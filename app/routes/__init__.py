from fastapi import FastAPI

from app.routes import chat, health


def register_routes(app: FastAPI) -> None:
    """Attach route modules to the application."""
    app.include_router(health.router)
    app.include_router(chat.router)
