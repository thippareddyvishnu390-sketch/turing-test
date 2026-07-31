from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings and environment variables configuration.
    
    Uses pydantic-settings to load variables from the environment 
    or a .env file. Validation is performed automatically on instantiation.
    """
    
    # Project Settings
    PROJECT_NAME: str = "AI Turing Test Workshop"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Gemini Configuration
    # Field validation ensures the key is present in the environment
    GEMINI_API_KEY: str = Field(..., min_length=1)
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"  # Using 1.5 Flash as requested (2.0/2.5 updates accordingly)

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Creates and caches a Settings instance.
    
    Using @lru_cache ensures that the .env file is read only once, 
    improving performance across multiple calls in the dependency injection system.
    
    Returns:
        Settings: The application configuration instance.
    """
    return Settings()