from http import HTTPStatus

from starlette.testclient import TestClient

from service.models import ModelNames
from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
    api_key: str,
) -> None:
    user_id = 123
    model_name = ModelNames.TEST_MODEL.value
    path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = client.get(path, headers=headers)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_model_not_found(
    client: TestClient,
    api_key: str,
):
    """Тест проверяет, что при запросе несуществующей модели возвращается ошибка ModelNotFoundError"""
    invalid_model_name = "non_existent_model"
    user_id = 123

    path = GET_RECO_PATH.format(model_name=invalid_model_name, user_id=user_id)
    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.get(path, headers=headers)

    assert response.status_code == 404
    assert response.json() == {
        "errors": [
            {
                "error_key": "model_not_found",
                "error_loc": None,
                "error_message": f"Model '{invalid_model_name}' not found",
            }
        ]
    }
