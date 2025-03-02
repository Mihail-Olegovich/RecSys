import time

from fastapi import FastAPI, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response

from service.log import access_logger, app_logger
from service.models import Error
from service.response import server_error
from service.settings import get_config


class AccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started_at = time.perf_counter()
        response = await call_next(request)
        request_time = time.perf_counter() - started_at

        status_code = response.status_code

        access_logger.info(
            msg="",
            extra={
                "request_time": round(request_time, 4),
                "status_code": status_code,
                "requested_url": request.url,
                "method": request.method,
            },
        )
        return response


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except HTTPException as he:
            app_logger.warning(f"HTTP exception occurred: {he.detail}")
            error = Error(error_key="auth_error", error_message=he.detail)
            return JSONResponse(
                status_code=he.status_code,
                content={"errors": [error.dict()]},
            )
        except Exception as e:
            app_logger.exception(msg=f"Caught unhandled {e.__class__} exception: {e}")
            error = Error(error_key="server_error", error_message="Internal Server Error")
            return server_error([error])


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        config = get_config()
        expected_api_key = config.auth_config.api_key

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            app_logger.warning("Missing Authorization header")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing API key",
            )

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            app_logger.warning("Invalid Authorization header format")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key format",
            )

        if parts[1] != expected_api_key:
            app_logger.warning("Invalid API key")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        return await call_next(request)


def add_middlewares(app: FastAPI) -> None:
    # do not change order
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(AccessMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
