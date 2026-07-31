from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["System"])


class HealthCheckResponse(BaseModel):
    """
    Schema for health check response.
    """
    status: str
    service: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Endpoint to verify the health and availability of the service.
    
    Returns:
        HealthCheckResponse: The current status and service identifier.
    """
    return HealthCheckResponse(
        status="healthy", 
        service="AI Turing Backend"
    )