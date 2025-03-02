from http import HTTPStatus

from starlette.testclient import TestClient


def test_access_with_valid_api_key(client: TestClient, api_key: str) -> None:
    """Тест проверяет, что запрос с правильным API-ключом выполняется успешно"""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.get("/reco/test_model/123", headers=headers)

    assert response.status_code == HTTPStatus.OK
    assert "user_id" in response.json()


def test_access_with_invalid_api_key(client: TestClient) -> None:
    """Тест проверяет, что запрос с неправильным API-ключом возвращает 403"""
    headers = {"Authorization": "Bearer invalid_key"}
    response = client.get("/reco/test_model/123", headers=headers)

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {
        "errors": [
            {
                "error_key": "auth_error",
                "error_loc": None,
                "error_message": "Invalid API key",
            }
        ]
    }


def test_access_without_api_key(client: TestClient) -> None:
    """Тест проверяет, что запрос без API-ключа возвращает 403"""
    response = client.get("/reco/test_model/123")

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {
        "errors": [
            {
                "error_key": "auth_error",
                "error_loc": None,
                "error_message": "Missing API key",
            }
        ]
    }


def test_invalid_auth_header_format(client: TestClient) -> None:
    """Тест проверяет, что запрос с неправильным форматом заголовка возвращает 403"""
    headers = {"Authorization": "some_key"}
    response = client.get("/reco/test_model/123", headers=headers)

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {
        "errors": [
            {
                "error_key": "auth_error",
                "error_loc": None,
                "error_message": "Invalid API key format",
            }
        ]
    }
