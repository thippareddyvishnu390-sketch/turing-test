import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import register_routes
from app.services.chat_service import ChatService
from app.utils.logging import (
    get_logger,
    log_api_failure,
    log_request_completed,
    log_request_started,
    log_shutdown,
    log_startup,
    setup_logging,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    try:
        chat_service = ChatService()
        chat_service.initialize()
        app.state.chat_service = chat_service
        app.state.settings = get_settings()
        log_startup(
            logger,
            component="app",
            chat_service_initialized=True,
            environment=app.state.settings.environment,
        )
        yield
    except Exception:
        logger.exception("Application startup failed")
        raise
    finally:
        log_shutdown(logger, component="app")


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug or settings.environment != "production" else None,
        redoc_url="/redoc" if settings.debug or settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.debug or settings.environment != "production" else None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.CORS_ORIGINS == "*" else [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4()}"
        started_at = time.perf_counter()
        log_request_started(
            logger,
            request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            log_api_failure(
                logger,
                request_id,
                "request failed",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

        duration_ms = (time.perf_counter() - started_at) * 1000
        log_request_completed(
            logger,
            request_id,
            duration_ms,
            status_code=response.status_code,
        )

        if response.status_code >= 400:
            logger.warning(
                "request completed with error status",
                extra={
                    "event": "request_failed",
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 3),
                },
            )

        return response

    register_routes(application)
    return application


app = create_app()
