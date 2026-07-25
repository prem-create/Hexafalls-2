"""
Walking Eye - AI Perception Engine
Model Manager.

Responsible for loading, holding, and providing access to the YOLO model.
The model is loaded ONCE at application startup and reused for every request.
This is the single source of truth for the AI model lifecycle.
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from app.utilities.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manages the lifecycle of the YOLO object detection model.

    Designed to be instantiated once and stored in app.state.
    Thread-safe for read operations (inference); YOLO handles this internally.

    Future extension: add depth model, OCR model, etc. as additional
    attributes following the same load-once pattern.
    """

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path: Path to the YOLO model weights (.pt file).
                        If the file doesn't exist, Ultralytics will
                        auto-download it from the official model hub.
        """
        self.model_path = model_path
        self._model: Optional[YOLO] = None

    def load(self) -> None:
        """
        Loads the YOLO model into memory.
        Called once during application startup via the lifespan handler.

        Raises:
            RuntimeError: If the model fails to load.
        """
        logger.info(f"Loading YOLO model from: {self.model_path}")

        model_file = Path(self.model_path)

        # Auto-download if not present (Ultralytics handles this natively)
        if not model_file.exists():
            logger.warning(
                f"Model file not found at '{self.model_path}'. "
                "Ultralytics will attempt to download it automatically."
            )

        try:
            self._model = YOLO(self.model_path)
            # Warm up the model with a dummy inference pass to avoid
            # cold-start latency on the first real request.
            self._warmup()
            logger.info(f"YOLO model loaded successfully: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _warmup(self) -> None:
        """
        Runs a silent dummy inference to initialize CUDA/CPU kernels.
        This ensures the first real request doesn't pay the warmup cost.
        """
        try:
            import numpy as np
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy_image, verbose=False)
            logger.info("Model warmup complete.")
        except Exception as e:
            # Warmup failure is non-fatal — log and continue
            logger.warning(f"Model warmup skipped due to error: {e}")

    @property
    def model(self) -> YOLO:
        """
        Returns the loaded YOLO model instance.

        Raises:
            RuntimeError: If accessed before load() has been called.
        """
        if self._model is None:
            raise RuntimeError(
                "YOLO model is not loaded. "
                "Ensure ModelManager.load() is called during startup."
            )
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is loaded and ready."""
        return self._model is not None

    def get_model_info(self) -> dict:
        """
        Returns basic model metadata for health checks and API responses.
        """
        if not self.is_loaded:
            return {"loaded": False, "path": self.model_path}

        return {
            "loaded": True,
            "path": self.model_path,
            "type": "YOLOv8",
            "task": getattr(self._model, "task", "detect"),
        }
