from rectools.models.popular import PopularModel
import pickle
import joblib
import os
import scipy
from implicit.nearest_neighbours import ItemItemRecommender
from rectools import Columns
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix
import numpy as np
from collections import Counter
from typing import Union
from rectools.dataset import Dataset

class UserKnn:
    """
    A user-based KNN model wrapper around `implicit.nearest_neighbours.ItemItemRecommender`
    """

    SIMILAR_USER_COLUMN = "similar_user_id"
    SIMILARITY_COLUMN = "similarity"
    IDF_COLUMN = "idf"

    def __init__(self, model: ItemItemRecommender, N_similar_users: int):
        self.model = model
        self.pop = PopularModel()
        self.N_similar_users = N_similar_users

        self.users_inv_mapping = None
        self.users_mapping = None
        self.items_inv_mapping = None
        self.items_mapping = None

        self.watched_items_dataframe = None
        self.item_idf = None
        self.cold_model_fitted = False

    def _set_mappings(self, interactions: pd.DataFrame) -> None:
        """
        Create dictionaries to map external IDs (users, items) to internal IDs and vice versa.
        """
        unique_users = interactions[Columns.User].unique()
        self.users_inv_mapping = dict(enumerate(unique_users))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        unique_items = interactions[Columns.Item].unique()
        self.items_inv_mapping = dict(enumerate(unique_items))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def _get_user_item_matrix(self, interactions: pd.DataFrame) -> spmatrix:
        """
        Construct a sparse user-item matrix in CSR format.
        Rows represent users, and columns represent items.
        """
        user_idx = interactions[Columns.User].map(self.users_mapping.get)
        item_idx = interactions[Columns.Item].map(self.items_mapping.get)
        data = interactions[Columns.Weight].astype(np.float32)

        user_item_coo = coo_matrix((data, (user_idx, item_idx)))
        return user_item_coo.tocsr()

    def _set_interacted_items_dataframe(self, interactions: pd.DataFrame) -> None:
        """
        Groups interactions by user to get item_id list for each user
        """
        self.interacted_items_dataframe = (
            interactions.groupby(Columns.User, as_index=False)
            .agg({Columns.Item: list})
            .rename(columns={Columns.User: self.SIMILAR_USER_COLUMN})
        )

    @staticmethod
    def idf(n: int, x: float):
        """
        Calculates IDF for one item
        """
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, interactions: pd.DataFrame) -> None:
        """
        Calculate IDF values for all items present in the interactions dataset
         and store the result in self.item_idf.
        """
        item_freqs = Counter(interactions[Columns.Item].values)
        item_idf_df = (
            pd.DataFrame
            .from_dict(item_freqs, orient="index", columns=["doc_freq"])
            .reset_index()
        )
        total_interactions = len(interactions)
        item_idf_df[self.IDF_COLUMN] = item_idf_df["doc_freq"].apply(
            lambda x: self.idf(total_interactions, x)
        )
        self.item_idf = item_idf_df

    def _prepare_for_model(self, train_interactions: pd.DataFrame) -> None:
        """
        Sets mappings, grouped interactions, calculates idf
        """
        self._set_mappings(train_interactions)
        self._set_interacted_items_dataframe(train_interactions)
        self._count_item_idf(train_interactions)

    def fit_cold_model(self, train_interactions: pd.DataFrame) -> None:
        """
        Fit a model for cold recommendations.

        Parameters:
        train_interactions (pd.DataFrame): interaction data used to train the model.
        """
        
        self.dataset_cold = Dataset.construct(
            interactions_df=train_interactions,
            user_features_df=None,
            item_features_df=None
        )
        
        self.pop.fit(self.dataset_cold)

    def recommend_cold(self, users: Union[list, np.array],
                        k: int = 100) -> pd.DataFrame:
        """
        Return recommendations for the given cold users.
        Can be called separately or within the class. Supports both list and numpy array as input.

        Parameters:
        users (list | np.array): List or array of users for whom recommendations will be generated.
        k (int, optional): Number of recommendations to generate per user. Default is 100.

        Returns:
        pd.DataFrame: A dataframe containing user-item recommendations.
        """
        
        pop_recs = self.pop.recommend(
            users,
            dataset=self.dataset_cold,
            k=k,
            filter_viewed=False  # True - удаляет просмотренные айтемы из рекомендаций 
        )

        return pop_recs

    def fit(self, train_interactions: pd.DataFrame) -> None:
        """
        Fit the model on the provided training data.

        Internally:
        1) Prepare mappings, watchlist DataFrame, and item IDF.
        2) Create a user-item matrix and fit the underlying Implicit model.
        """
        self.fit_cold_model(train_interactions)
        self._prepare_for_model(train_interactions)
        user_item_matrix = self._get_user_item_matrix(train_interactions)
        user_item_matrix = user_item_matrix.astype(np.float64)
        self.model.fit(user_item_matrix.T)

    def _get_similar_users(self, external_user_id: int) -> tuple[list[int], list[float]]:
        """
        Retrieve a list of similar users and corresponding similarities
        from the underlying Implicit model.
        """
        if external_user_id not in self.users_mapping:
            # if user doesn't exist in mapping, return sentinel (-1).
            return [-1], [-1]

        internal_user_id = self.users_mapping[external_user_id]
        user_ids, similarities = self.model.similar_items(
            internal_user_id,
            N=self.N_similar_users
        )
        # convert back to external IDs
        external_user_ids = [self.users_inv_mapping[u_id] for u_id in user_ids]
        return external_user_ids, similarities

    @staticmethod
    def get_rank(recs: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Sort recommendations by score in descending order,
        assign ranks within each user group, and then truncate by top-k.
        """
        recs = recs.sort_values([Columns.User, Columns.Score], ascending=False)
        recs = recs.drop_duplicates([Columns.User, Columns.Item])
        recs[Columns.Rank] = recs.groupby(Columns.User).cumcount() + 1
        recs = recs[recs[Columns.Rank] <= k][
            [Columns.User, Columns.Item, Columns.Score, Columns.Rank]
        ]

        return recs

    def recommend(self, users: np.ndarray, k: int) -> pd.DataFrame:
        """
        Generate top-k recommendations for the specified list of users.
        """
        # Создаем начальный DataFrame с пользователями
        recs = pd.DataFrame({Columns.User: users})
        
        # Получаем похожих пользователей и их сходство
        recs[self.SIMILAR_USER_COLUMN], recs[self.SIMILARITY_COLUMN] = zip(
            *recs[Columns.User].map(lambda user_id: self._get_similar_users(user_id))
        )
        recs = recs.set_index(Columns.User).apply(pd.Series.explode).reset_index()

        # Фильтруем и обрабатываем рекомендации на основе похожих пользователей
        knn_recs = (
            recs[~(recs[Columns.User] == recs[self.SIMILAR_USER_COLUMN])]
            .merge(
                self.interacted_items_dataframe,
                on=[self.SIMILAR_USER_COLUMN],
                how="left",
            )
            .explode(Columns.Item)
            .sort_values([Columns.User, self.SIMILARITY_COLUMN], ascending=False)
            .drop_duplicates([Columns.User, Columns.Item], keep="first")
            .merge(self.item_idf, left_on=Columns.Item, right_on="index", how="left")
        )

        knn_recs[Columns.Score] = knn_recs[self.SIMILARITY_COLUMN] * knn_recs[self.IDF_COLUMN]
        knn_recs = knn_recs[[Columns.User, Columns.Item, Columns.Score]]
        knn_recs = knn_recs.dropna()

        # Сохраняем полный список пользователей для проверки в конце
        all_users = set(users)
        
        # Пользователи, которые получили KNN рекомендации
        users_with_knn_recs = set(knn_recs[Columns.User].unique())
        
        # Пользователи, которым не хватает KNN рекомендаций до k 
        user_rec_counts = knn_recs[Columns.User].value_counts().reset_index()
        user_rec_counts.columns = [Columns.User, 'count']
        users_needing_more = user_rec_counts[user_rec_counts['count'] < k][Columns.User].tolist()
        
        # Пользователи, у которых нет KNN рекомендаций вообще
        users_without_knn_recs = all_users - users_with_knn_recs
        
        # Объединяем обе категории пользователей, которым нужны популярные рекомендации
        users_needing_pop_recs = list(users_without_knn_recs) + users_needing_more
        
        if users_needing_pop_recs:
            # Получаем популярные рекомендации для всех пользователей, которым они нужны
            pop_recs_df = self.pop.recommend(
                users=users_needing_pop_recs,
                dataset=self.dataset_cold,
                k=k,  # Запрашиваем максимальное количество
                filter_viewed=False
            )
            
            # Для пользователей без KNN рекомендаций - берем все популярные рекомендации
            pop_only_users = pop_recs_df[pop_recs_df[Columns.User].isin(users_without_knn_recs)]
            
            # Для пользователей с неполными KNN рекомендациями - фильтруем по необходимости
            if users_needing_more:
                # Получаем уже рекомендованные элементы для каждого пользователя
                existing_items_by_user = knn_recs.groupby(Columns.User)[Columns.Item].apply(set).to_dict()
                
                # Обрабатываем пользователей с неполными рекомендациями
                additional_recs = []
                for user_id in users_needing_more:
                    items_needed = k - len(existing_items_by_user.get(user_id, set()))
                    user_pop_recs = pop_recs_df[pop_recs_df[Columns.User] == user_id]
                    user_pop_recs = user_pop_recs[~user_pop_recs[Columns.Item].isin(existing_items_by_user.get(user_id, set()))]
                    user_pop_recs = user_pop_recs.head(items_needed)
                    additional_recs.append(user_pop_recs)
                
                # Объединяем дополнительные рекомендации если они есть
                if additional_recs:
                    additional_recs_df = pd.concat(additional_recs, ignore_index=True)
                    # Объединяем все рекомендации
                    recs = pd.concat([knn_recs, pop_only_users, additional_recs_df], ignore_index=True)
                else:
                    recs = pd.concat([knn_recs, pop_only_users], ignore_index=True)
            else:
                recs = pd.concat([knn_recs, pop_only_users], ignore_index=True)
        else:
            recs = knn_recs
        
        # Получаем итоговые ранжированные рекомендации
        recs = self.get_rank(recs, k=k)
        
        # Проверяем, что все пользователи получили рекомендации
        final_users = set(recs[Columns.User].unique())
        missing_users = all_users - final_users
        
        # Если остались пользователи без рекомендаций, добавляем им популярные рекомендации напрямую
        if missing_users:
            last_chance_recs = self.pop.recommend(
                users=list(missing_users),
                dataset=self.dataset_cold,
                k=k,
                filter_viewed=False
            )
            recs = pd.concat([recs, last_chance_recs], ignore_index=True)
            recs = self.get_rank(recs, k=k)
        
        return recs
    


    def save(self, path):
        """
        Сохраняет модель в указанную директорию.
        
        Parameters:
        path (str): Путь к директории, куда будет сохранена модель
        """
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем основные параметры модели
        model_params = {
            'N_similar_users': self.N_similar_users,
            'users_inv_mapping': self.users_inv_mapping,
            'users_mapping': self.users_mapping,
            'items_inv_mapping': self.items_inv_mapping,
            'items_mapping': self.items_mapping,
            'cold_model_fitted': self.cold_model_fitted
        }
        
        with open(os.path.join(path, 'model_params.pkl'), 'wb') as f:
            pickle.dump(model_params, f)
        
        # Сохраняем датафреймы
        if hasattr(self, 'interacted_items_dataframe') and self.interacted_items_dataframe is not None:
            self.interacted_items_dataframe.to_pickle(os.path.join(path, 'interacted_items_df.pkl'))
        
        if hasattr(self, 'item_idf') and self.item_idf is not None:
            self.item_idf.to_pickle(os.path.join(path, 'item_idf.pkl'))
        
        # Сохраняем матрицу схожести и другие компоненты модели ItemItemRecommender
        if hasattr(self.model, 'similarity') and self.model.similarity is not None:
            scipy.sparse.save_npz(os.path.join(path, 'similarity_matrix.npz'), self.model.similarity)
        
        # Сохраняем дополнительные параметры модели ItemItemRecommender
        if hasattr(self.model, 'K'):
            item_model_params = {
                'K': self.model.K,
                'num_threads': getattr(self.model, 'num_threads', 0),
                'filter_users': getattr(self.model, 'filter_users', True),
                'filter_items': getattr(self.model, 'filter_items', True),
                'sort_items': getattr(self.model, 'sort_items', True)
            }
            with open(os.path.join(path, 'item_model_params.pkl'), 'wb') as f:
                pickle.dump(item_model_params, f)
        
        # Сохраняем популярную модель и датасет для холодного старта
        if hasattr(self, 'pop') and self.pop is not None:
            joblib.dump(self.pop, os.path.join(path, 'popular_model.joblib'))
        
        if hasattr(self, 'dataset_cold') and self.dataset_cold is not None:
            joblib.dump(self.dataset_cold, os.path.join(path, 'dataset_cold.joblib'))


    @classmethod
    def load(cls, path, implicit_model=None):
        """
        Загружает модель из указанной директории.
        
        Parameters:
        path (str): Путь к директории с сохраненной моделью
        implicit_model: Экземпляр класса ItemItemRecommender (если None, будет создан новый)
        
        Returns:
        UserKnn: Экземпляр загруженной модели
        """
            
        if implicit_model is None:
            from implicit.nearest_neighbours import ItemItemRecommender
            implicit_model = ItemItemRecommender()
        
        # Если implicit_model все еще None, создаем с параметрами по умолчанию
        if implicit_model is None:
            from implicit.nearest_neighbours import ItemItemRecommender
            implicit_model = ItemItemRecommender()
        
        # Загружаем матрицу схожести, если она существует
        if os.path.exists(os.path.join(path, 'similarity_matrix.npz')):
            implicit_model.similarity = scipy.sparse.load_npz(os.path.join(path, 'similarity_matrix.npz'))
        
        # Создаем пустой экземпляр класса
        with open(os.path.join(path, 'model_params.pkl'), 'rb') as f:
            model_params = pickle.load(f)
        
        instance = cls(model=implicit_model, N_similar_users=model_params['N_similar_users'])
        
        # Загружаем основные параметры
        for key, value in model_params.items():
            if key != 'N_similar_users':  # N_similar_users уже установлен в конструкторе
                setattr(instance, key, value)
        
        # Загружаем датафреймы
        if os.path.exists(os.path.join(path, 'interacted_items_df.pkl')):
            instance.interacted_items_dataframe = pd.read_pickle(os.path.join(path, 'interacted_items_df.pkl'))
        
        if os.path.exists(os.path.join(path, 'item_idf.pkl')):
            instance.item_idf = pd.read_pickle(os.path.join(path, 'item_idf.pkl'))
        
        # Загружаем популярную модель и датасет для холодного старта
        if os.path.exists(os.path.join(path, 'popular_model.joblib')):
            instance.pop = joblib.load(os.path.join(path, 'popular_model.joblib'))
        
        if os.path.exists(os.path.join(path, 'dataset_cold.joblib')):
            instance.dataset_cold = joblib.load(os.path.join(path, 'dataset_cold.joblib'))
        
        return instance       