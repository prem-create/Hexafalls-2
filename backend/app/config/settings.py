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
    ENABLE_DEPTH: bool = True
    ENABLE_TRACKING: bool = True
    EXTERNAL_LLM_API_KEY: str = ""
    EXTERNAL_LLM_ENDPOINT: str = ""

    # --- Depth Estimation (MiDaS DPT_Hybrid) ---
    # MiDaS model type: "DPT_Hybrid" | "DPT_Large" | "MiDaS_small"
    DEPTH_MODEL_TYPE: str = "DPT_Hybrid"

    # Fixed scene-level scale factor: converts MiDaS unitless output → metres.
    # Tuned for typical indoor/outdoor walking scenes at 1–10 m range.
    # Override via .env if your environment needs a different value.
    DEPTH_SCALE_FACTOR: float = 7.5

    # When a person is visible, use their bbox height + known average height
    # to auto-calibrate the scale factor each frame (person-anchored mode).
    # Set False to use DEPTH_SCALE_FACTOR unconditionally.
    DEPTH_PERSON_ANCHOR: bool = True

    # Assumed average person height in metres for person-anchored calibration.
    PERSON_HEIGHT_M: float = 1.70

    # Depth sampling patch size around each detection's centre (pixels).
    # The median of this NxN patch is used — reduces boundary noise.
    DEPTH_PATCH_SIZE: int = 11

    # --- Temporal Motion Analysis ---
    # Number of past observations kept per tracked object.
    HISTORY_SIZE: int = 5

    # Minimum observations required before issuing a motion classification.
    MIN_TRACK_HISTORY: int = 2

    # Depth-based thresholds (metres). Change smaller than STATIONARY is
    # treated as STATIONARY; change larger than APPROACHING threshold is
    # treated as APPROACHING or MOVING_AWAY.
    STATIONARY_DEPTH_THRESHOLD: float = 0.08   # metres
    APPROACHING_DEPTH_THRESHOLD: float = 0.08  # metres (same as stationary edge)

    # BBox-scale thresholds (fractional area change, 0–1).
    # E.g. 0.05 means a 5 % change in bbox area is still considered stationary.
    STATIONARY_SCALE_THRESHOLD: float = 0.05
    APPROACHING_SCALE_THRESHOLD: float = 0.05

    # Pixel threshold below which centre-point movement is ignored as noise.
    DIRECTION_NOISE_THRESHOLD_PX: int = 5

    # Minimum confidence to accept a motion classification.
    MOTION_CONFIDENCE_THRESHOLD: float = 0.50

    # Tracker: IoU overlap required to associate a detection with an existing track.
    TRACKER_IOU_THRESHOLD: float = 0.30

    # How many frames a track may go unmatched before it is removed.
    TRACKER_MAX_AGE: int = 5

    # Maximum number of concurrent tracking sessions held in memory.
    # Each session has its own tracker + history buffers.
    MAX_TRACKER_SESSIONS: int = 50

    # --- Alert Manager ---
    # Minimum seconds between repeated alerts for the same track_id.
    # Prevents the same message from firing on every frame.
    ALERT_MIN_INTERVAL_S: float = 3.0

    # Distance zone boundaries in metres (used for zone-crossing alerts).
    # An alert fires when an object crosses from one zone to another.
    ALERT_ZONE_FAR_M: float = 5.0        # beyond this → FAR
    ALERT_ZONE_MEDIUM_M: float = 3.0     # beyond this → MEDIUM
    ALERT_ZONE_NEAR_M: float = 1.5       # beyond this → NEAR
    # below ALERT_ZONE_NEAR_M → VERY_NEAR

    # Minimum distance change (metres) that counts as meaningful.
    # Changes smaller than this within the same zone are suppressed.
    ALERT_DISTANCE_CHANGE_THRESHOLD_M: float = 0.4

    # How many m/s counts as "rapid" approach (triggers high-priority alert).
    ALERT_RAPID_APPROACH_THRESHOLD_MS: float = 1.5

    # Number of frames a track can be absent before its alert state is reset.
    ALERT_TRACK_DISAPPEAR_FRAMES: int = 10

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
