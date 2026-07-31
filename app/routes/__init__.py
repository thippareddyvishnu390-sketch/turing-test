from fastapi import FastAPI

from app.routes import health


def register_routes(app: FastAPI) -> None:
    """Attach route modules to the application."""
    app.include_router(health.router)
