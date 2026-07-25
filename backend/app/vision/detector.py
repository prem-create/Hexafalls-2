"""
Walking Eye - AI Perception Engine
Object Detector.

Wraps Ultralytics YOLO inference and converts raw results
into clean internal data structures.

This layer knows nothing about HTTP, Pydantic schemas, or the service layer.
It only speaks: numpy array in → list of detections out.
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from app.utilities.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """
    Internal representation of a single detected object.

    Intentionally separate from the Pydantic schema so the
    detector stays decoupled from the API contract.
    """

    label: str
    confidence: float
    x: int          # Bounding box left edge (pixels)
    y: int          # Bounding box top edge (pixels)
    width: int      # Bounding box width (pixels)
    height: int     # Bounding box height (pixels)
    center_x: int
    center_y: int


class DetectionError(Exception):
    """Raised when YOLO inference fails unexpectedly."""
    pass


class ObjectDetector:
    """
    Runs YOLO inference on a preprocessed image array.

    Receives the YOLO model via constructor injection —
    the model is never loaded here, only used.
    """

    def __init__(self, model: YOLO, confidence_threshold: float = 0.45) -> None:
        """
        Args:
            model: A loaded Ultralytics YOLO instance.
            confidence_threshold: Minimum confidence to include a detection.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> tuple[List[Detection], float]:
        """
        Runs inference on the provided image array.

        Args:
            image: BGR or RGB numpy array (H x W x 3).

        Returns:
            Tuple of (list of Detection objects, inference_time_ms).

        Raises:
            DetectionError: If YOLO inference throws an unexpected error.
        """
        logger.info(
            f"Running inference | image shape: {image.shape} | "
            f"confidence threshold: {self.confidence_threshold}"
        )

        try:
            start = time.perf_counter()

            results = self.model(
                image,
                conf=self.confidence_threshold,
                verbose=False,   # Suppress Ultralytics stdout noise
            )

            inference_ms = (time.perf_counter() - start) * 1000
            logger.info(f"Inference complete in {inference_ms:.1f} ms")

        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise DetectionError(f"Object detection failed: {e}") from e

        detections = self._parse_results(results)
        logger.info(f"Detected {len(detections)} object(s)")

        return detections, round(inference_ms, 2)

    def _parse_results(self, results) -> List[Detection]:
        """
        Converts raw Ultralytics Results into Detection dataclasses.

        Args:
            results: List of Ultralytics Result objects.

        Returns:
            List of Detection instances.
        """
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):
                try:
                    # xyxy format: x1, y1, x2, y2
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    confidence = float(boxes.conf[i])
                    class_id = int(boxes.cls[i])
                    label = result.names.get(class_id, f"class_{class_id}")

                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)

                    detections.append(
                        Detection(
                            label=label,
                            confidence=round(confidence, 4),
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            center_x=cx,
                            center_y=cy,
                        )
                    )

                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping malformed detection at index {i}: {e}")
                    continue

        return detections
