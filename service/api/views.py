from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError
from service.api.get_ann_reco import get_recommendations
from service.log import app_logger
from service.models import Error, ModelNames


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        404: {
            "model": Error,
            "description": "Model not found",
            "content": {
                "application/json": {
                    "example": {
                        "errors": [
                            {
                                "error_key": "model_not_found",
                                "error_message": "Model 'unknown_model' not found",
                                "error_loc": None,
                            }
                        ]
                    }
                }
            },
        }
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name not in [model.value for model in ModelNames]:
        raise ModelNotFoundError(
            error_message=f"Model '{model_name}' not found",
        )

    k_recs = request.app.state.k_recs
    reco = get_recommendations(user_id, k_recs, request)

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
