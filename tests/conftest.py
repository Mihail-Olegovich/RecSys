# pylint: disable=redefined-outer-name
import os
from unittest.mock import Mock, patch

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
def app(service_config: ServiceConfig) -> FastAPI:
    # Создаём фейковый DataFrame, который будет возвращать метод recommend_cold
    mock_df = pd.DataFrame(
        {
            "user_id": [123] * service_config.k_recs,
            "item_id": list(range(1, service_config.k_recs + 1)),
            "score": [0.9] * service_config.k_recs,
            "rank": list(range(1, service_config.k_recs + 1)),
        }
    )

    # Создаём мок для модели
    mock_model = Mock()
    mock_model.recommend_cold.return_value = mock_df

    # Патчим метод load, чтобы он возвращал мок вместо реальной модели
    with patch("service.api.userknn.UserKnn.load", return_value=mock_model):
        app = create_app(service_config)

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app=app)


@pytest.fixture
def api_key() -> str:
    """API-ключ для тестов"""
    return os.environ.get("AUTH_API_KEY")
