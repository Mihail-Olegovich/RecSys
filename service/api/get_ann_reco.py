from typing import List

from fastapi import Request


def recommend_cold(user_id: int, k: int = 100, request: Request = None) -> List[int]:
    pop_recs = request.app.state.pop.recommend(
        [user_id],
        dataset=request.app.state.dataset_cold,
        k=k,
        filter_viewed=True,
    )

    return pop_recs.sort_values(by="rank")["item_id"].tolist()


def get_recommendations(user_id: int, k: int = 10, request: Request = None) -> List[int]:
    if user_id not in request.app.state.user_id_to_idx:
        return recommend_cold(user_id, k, request)

    user_idx = request.app.state.user_id_to_idx[user_id]

    user_emb = request.app.state.user_embeddings[[user_idx], :]

    labels, _ = request.app.state.hnsw.knn_query(user_emb, k=k)

    recommended_items = [request.app.state.idx_to_item_id[label] for label in labels[0]]

    return recommended_items
