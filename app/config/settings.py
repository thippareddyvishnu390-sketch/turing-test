from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ENV_FILE: Final[Path] = Path(".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    PROJECT_NAME: str = "AI Turing Test Workshop"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    APP_NAME: str = "Turing Test Chatbot"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "Turing Test chatbot API."
    CORS_ORIGINS: str = "*"
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    GEMINI_API_KEY: str = Field(..., min_length=1)
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"
    GEMINI_TEMPERATURE: float = Field(default=0.8, ge=0.0, le=2.0)
    GEMINI_MAX_OUTPUT_TOKENS: int | None = Field(default=None, ge=1)

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    @property
    def app_name(self) -> str:
        return self.APP_NAME

    @property
    def app_version(self) -> str:
        return self.APP_VERSION

    @property
    def app_description(self) -> str:
        return self.APP_DESCRIPTION

    @property
    def debug(self) -> bool:
        return self.DEBUG

    @property
    def environment(self) -> str:
        return self.ENVIRONMENT

    @field_validator("GEMINI_MODEL_NAME")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("GEMINI_MODEL_NAME must not be empty.")
        return cleaned_value

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        return value.strip().lower() or "development"


@lru_cache
def get_settings() -> Settings:
    """Create and cache the application settings instance."""
    return Settings()