import typing as tp
from enum import Enum

from pydantic import BaseModel


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class ModelNames(str, Enum):
    TEST_MODEL = "test_model"
    USER_KNN = 'UserKnn'
