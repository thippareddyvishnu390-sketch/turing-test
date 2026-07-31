from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIError(BaseModel):
    """
    Standardized API error response schema.
    """
    message: str = Field(..., description="A human-readable explanation of the error.")
    code: str = Field(..., description="A machine-readable error code.")
    details: Any | None = Field(None, description="Optional granular details about the error.")


class SuccessResponse(BaseModel, Generic[T]):
    """
    Standardized success response schema for generic data.
    """
    success: bool = Field(default=True)
    message: str | None = Field(None, description="Optional success message.")
    data: T | None = Field(None, description="The payload of the response.")


class HealthStatus(BaseModel):
    """
    Schema for application health check information.
    """
    status: str = Field(default="ok", description="The current status of the service.")
    version: str = Field(..., description="The application version.")
    environment: str = Field(..., description="The running environment.")