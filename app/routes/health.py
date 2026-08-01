from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    version = getattr(settings, "app_version", None) or getattr(settings, "APP_VERSION", "0.1.0")
    environment = getattr(settings, "environment", None) or getattr(settings, "ENVIRONMENT", "development")

    return HealthResponse(
        status="ok",
        version=version,
        environment=environment,
    )