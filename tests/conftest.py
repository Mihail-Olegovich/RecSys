# pylint: disable=redefined-outer-name
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from service.api.app import create_app
from service.settings import ServiceConfig, get_config


@pytest.fixture
def service_config() -> ServiceConfig:
    return get_config()


@pytest.fixture
def app(
    service_config: ServiceConfig,
) -> FastAPI:
    app = create_app(service_config)
    return app


@pytest.fixture
def api_key() -> str:
    """API-ключ для тестов"""
    return os.environ.get("AUTH_API_KEY")


@pytest.fixture
def mock_data():
    """Фикстура с моковыми данными для тестов"""

    test_user_id = 123
    test_user_idx = 0
    test_item_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

    user_id_to_idx = {test_user_id: test_user_idx}
    idx_to_user_id = {test_user_idx: test_user_id}

    item_id_to_idx = dict(enumerate(test_item_ids))
    idx_to_item_id = dict(enumerate(test_item_ids))

    user_embeddings = np.random.random((1, 64))
    item_embeddings = np.random.random((len(test_item_ids), 64))

    return {
        "user_id_to_idx": user_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "item_id_to_idx": item_id_to_idx,
        "idx_to_item_id": idx_to_item_id,
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "test_user_id": test_user_id,
        "test_item_ids": test_item_ids,
    }


@pytest.fixture
def mock_hnsw():
    """Мок для HNSW индекса"""
    hnsw_mock = MagicMock()

    def mock_knn_query(embedding, k):
        return np.array([list(range(k))]), np.random.random((1, k))

    hnsw_mock.knn_query = mock_knn_query
    return hnsw_mock


@pytest.fixture
def mock_pop_model():
    """Мок для популярной модели рекомендаций"""
    pop_mock = MagicMock()

    def mock_recommend(user_ids, dataset, k, filter_viewed):
        items = [1001 + i for i in range(k)]
        return pd.DataFrame({"user_id": [user_ids[0]] * k, "item_id": items, "rank": list(range(1, k + 1))})

    pop_mock.recommend = mock_recommend
    return pop_mock


@pytest.fixture
def mock_dataset_cold():
    """Мок для холодного датасета"""
    return MagicMock()


@pytest.fixture
def patched_app(app, mock_data, mock_hnsw, mock_pop_model, mock_dataset_cold):
    """Патчит приложение моковыми объектами"""
    app.state.user_id_to_idx = mock_data["user_id_to_idx"]
    app.state.idx_to_user_id = mock_data["idx_to_user_id"]
    app.state.item_id_to_idx = mock_data["item_id_to_idx"]
    app.state.idx_to_item_id = mock_data["idx_to_item_id"]
    app.state.user_embeddings = mock_data["user_embeddings"]
    app.state.item_embeddings = mock_data["item_embeddings"]
    app.state.hnsw = mock_hnsw
    app.state.pop = mock_pop_model
    app.state.dataset_cold = mock_dataset_cold

    return app


@pytest.fixture
def client(patched_app: FastAPI) -> TestClient:
    """Клиент для тестирования с моками"""
    return TestClient(app=patched_app)
