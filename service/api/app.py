import asyncio
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict

import hnswlib
import joblib
import numpy as np
import uvloop
from fastapi import FastAPI

from ..log import app_logger, setup_logging
from ..settings import ServiceConfig
from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .views import add_views

__all__ = ("create_app",)
PATH_TO_DATA = "/Users/kulyaskin_mikhail/ITMO/RecSys/data"


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)

    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs

    with open(PATH_TO_DATA + "/id_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        app.state.user_id_to_idx = mappings["user_id_to_idx"]
        app.state.item_id_to_idx = mappings["item_id_to_idx"]
        app.state.idx_to_user_id = mappings["idx_to_user_id"]
        app.state.idx_to_item_id = mappings["idx_to_item_id"]

    app.state.user_embeddings = np.load(PATH_TO_DATA + "/user_embeddings.npy")
    app.state.item_embeddings = np.load(PATH_TO_DATA + "/item_embeddings.npy")

    with open(PATH_TO_DATA + "/config.pkl", "rb") as f:
        config_params = pickle.load(f)
        hnsw_params = config_params["hnsw_params"]
    # pylint: disable=c-extension-no-member
    app.state.hnsw = hnswlib.Index(hnsw_params["space"], hnsw_params["dim"])
    app.state.hnsw.load_index(PATH_TO_DATA + "/hnsw_index.bin", max_elements=len(app.state.item_id_to_idx))
    app.state.hnsw.set_ef(hnsw_params["efS"])

    app.state.pop = joblib.load(PATH_TO_DATA + "/popular_model.joblib")
    app.state.dataset_cold = joblib.load(PATH_TO_DATA + "/dataset_cold.joblib")

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
