"""
Walking Eye - AI Perception Engine
Model Manager.

Responsible for loading, holding, and providing access to:
  - YOLO object detection model
  - MiDaS depth estimation model (when ENABLE_DEPTH=True)

Both models are loaded ONCE at application startup and reused for every
request.  This is the single source of truth for the AI model lifecycle.
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from app.utilities.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manages the lifecycle of the YOLO and (optionally) MiDaS depth models.

    Designed to be instantiated once and stored in app.state.
    Thread-safe for read operations (inference); YOLO and torch handle
    this internally.
    """

    def __init__(self, model_path: str, enable_depth: bool = False) -> None:
        """
        Args:
            model_path:    Path to the YOLO model weights (.pt file).
                           Ultralytics auto-downloads if not present.
            enable_depth:  When True, also loads MiDaS at startup.
        """
        self.model_path = model_path
        self.enable_depth = enable_depth

        self._model: Optional[YOLO] = None
        self._depth_estimator = None   # DepthEstimator | None

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Loads the YOLO model (and optionally the depth model) into memory.
        Called once during application startup via the lifespan handler.

        Raises:
            RuntimeError: If the YOLO model fails to load.
        """
        logger.info(f"Loading YOLO model from: {self.model_path}")

        if not Path(self.model_path).exists():
            logger.warning(
                f"Model file not found at '{self.model_path}'. "
                "Ultralytics will attempt to download it automatically."
            )

        try:
            self._model = YOLO(self.model_path)
            self._warmup_yolo()
            logger.info(f"YOLO model loaded successfully: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

        # --- Depth model ---
        if self.enable_depth:
            self._load_depth_model()

    def _warmup_yolo(self) -> None:
        """Silent YOLO warmup to initialise CUDA/CPU kernels."""
        try:
            import numpy as np
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            logger.info("YOLO warmup complete.")
        except Exception as e:
            logger.warning(f"YOLO warmup skipped: {e}")

    # ------------------------------------------------------------------
    # Depth (MiDaS)
    # ------------------------------------------------------------------

    def _load_depth_model(self) -> None:
        """Loads MiDaS depth estimator.  Non-fatal — logs and continues."""
        try:
            from app.config.settings import get_settings
            from app.vision.depth_estimator import DepthEstimator

            s = get_settings()
            estimator = DepthEstimator(
                model_type=s.DEPTH_MODEL_TYPE,
                scale_factor=s.DEPTH_SCALE_FACTOR,
                person_anchor=s.DEPTH_PERSON_ANCHOR,
                person_height_m=s.PERSON_HEIGHT_M,
                patch_size=s.DEPTH_PATCH_SIZE,
            )
            estimator.load()
            self._depth_estimator = estimator
            logger.info("Depth estimator loaded and ready.")
        except Exception as e:
            # Depth failure is non-fatal — fall back to bbox-proxy mode
            logger.warning(
                f"Depth estimator failed to load — "
                f"depth will be unavailable this session: {e}"
            )
            self._depth_estimator = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> YOLO:
        """Returns the loaded YOLO model instance."""
        if self._model is None:
            raise RuntimeError(
                "YOLO model is not loaded. "
                "Ensure ModelManager.load() is called during startup."
            )
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Returns True if the YOLO model is loaded and ready."""
        return self._model is not None

    @property
    def depth_estimator(self):
        """Returns the DepthEstimator instance, or None when not loaded."""
        return self._depth_estimator

    @property
    def depth_enabled(self) -> bool:
        """True when depth estimation is loaded and operational."""
        return self._depth_estimator is not None and self._depth_estimator.is_loaded

    def get_model_info(self) -> dict:
        """Returns model metadata for health checks and API responses."""
        info = {
            "yolo": {
                "loaded": self.is_loaded,
                "path": self.model_path,
                "type": "YOLO11n",
            },
            "depth": {
                "loaded": self.depth_enabled,
                "model_type": getattr(
                    self._depth_estimator, "model_type", None
                ),
            },
        }
        return info
