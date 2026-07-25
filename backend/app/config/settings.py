"""
Walking Eye - AI Perception Engine
Application settings and environment configuration.

Uses pydantic-settings for type-safe env var parsing.
All configuration is driven by environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object.
    Values are read from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # --- Model ---
    MODEL_PATH: str = "models/yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.45

    # --- Image Processing ---
    MAX_IMAGE_SIZE_MB: float = 10.0
    MAX_IMAGE_DIMENSION: int = 1280   # Longest side resized to this if exceeded
    ALLOWED_CONTENT_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    # --- Uploads ---
    UPLOAD_DIR: str = "uploads"

    # --- Future: OCR, Depth, LLM, etc. ---
    # These keys are defined now so future modules can read them
    # without changing the settings schema.
    ENABLE_OCR: bool = False
    ENABLE_DEPTH: bool = False
    ENABLE_TRACKING: bool = False
    EXTERNAL_LLM_API_KEY: str = ""
    EXTERNAL_LLM_ENDPOINT: str = ""

    @property
    def max_image_bytes(self) -> int:
        """Maximum allowed image size in bytes."""
        return int(self.MAX_IMAGE_SIZE_MB * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.
    Cached so the .env file is read only once per process.
    """
    return Settings()
