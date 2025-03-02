# pylint: disable=redefined-outer-name
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.testclient import TestClient

from service.api.app import create_app
from service.settings import ServiceConfig, get_config

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


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
def client(app: FastAPI) -> TestClient:
    return TestClient(app=app)


@pytest.fixture
def api_key() -> str:
    """API-ключ для тестов"""
    return os.environ.get("AUTH_API_KEY")
