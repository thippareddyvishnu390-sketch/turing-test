import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat, health
from app.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles application startup and shutdown events.
    """
    logger.info("Starting up AI Turing Backend...")
    # Initialize resources if necessary
    yield
    logger.info("Shutting down AI Turing Backend...")


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    """
    application = FastAPI(
        title="AI Turing Backend",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production to specific domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    application.include_router(health.router)
    application.include_router(chat.router)

    return application


app = create_app()