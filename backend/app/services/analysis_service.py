"""
Walking Eye - AI Perception Engine
Analysis Service.

Orchestrates the full perception pipeline:
  raw bytes → image processing → object detection → scene reasoning → response

This is the only place that knows about all three layers.
Routes call this service; they never touch vision or reasoning directly.
"""

import time
from typing import List

from ultralytics import YOLO

from app.config.settings import get_settings
from app.reasoning.scene_analyzer import SceneAnalyzer
from app.reasoning.spatial_analyzer import SpatialAnalyzer, SpatialDetection
from app.schemas.analysis import (
    AnalysisResponse,
    BoundingBox,
    Center,
    DetectedObject,
)
from app.utilities.logger import get_logger
from app.vision.detector import Detection, DetectionError, ObjectDetector
from app.vision.image_processor import ImageProcessingError, ImageProcessor

logger = get_logger(__name__)
settings = get_settings()


class AnalysisService:
    """
    Orchestrates image analysis from raw bytes to structured response.

    Dependencies are injected — no globals, no app.state access here.
    This makes the service fully testable without a running FastAPI app.
    """

    def __init__(self, model: YOLO) -> None:
        """
        Args:
            model: Loaded YOLO instance, injected from app.state via DI.
        """
        self._processor = ImageProcessor(
            max_dimension=settings.MAX_IMAGE_DIMENSION
        )
        self._detector = ObjectDetector(
            model=model,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        )
        self._spatial = SpatialAnalyzer()
        self._analyzer = SceneAnalyzer()

    async def analyze(self, image_bytes: bytes) -> AnalysisResponse:
        """
        Runs the full analysis pipeline on raw image bytes.

        Args:
            image_bytes: Raw bytes from the uploaded image file.

        Returns:
            AnalysisResponse with detections and scene summary.

        Raises:
            ImageProcessingError: If the image cannot be decoded or is invalid.
            DetectionError: If YOLO inference fails.
        """
        pipeline_start = time.perf_counter()

        # --- Step 1: Image Processing ---
        logger.info("Step 1/3: Processing image...")
        processed = self._processor.process(image_bytes)

        if processed.was_resized:
            logger.info(
                f"Image resized: {processed.original_width}x{processed.original_height}"
                f" → {processed.processed_width}x{processed.processed_height}"
            )

        # --- Step 2: Object Detection ---
        logger.info("Step 2/3: Running object detection...")
        detections, inference_ms = self._detector.detect(processed.array)

        logger.info(
            f"Detection complete: {len(detections)} object(s) found "
            f"in {inference_ms:.1f} ms"
        )

        # --- Step 3: Spatial Awareness ---
        logger.info("Step 3/4: Analyzing direction and proximity...")
        spatial_detections = self._spatial.analyze(
            detections,
            frame_width=processed.processed_width,
            frame_height=processed.processed_height,
        )

        # --- Step 4: Scene Reasoning ---
        logger.info("Step 4/4: Generating scene summary...")
        summary, suggested_direction = self._analyzer.summarize(
            spatial_detections,
            frame_width=processed.processed_width,
            frame_height=processed.processed_height,
        )

        # --- Build Response ---
        total_ms = (time.perf_counter() - pipeline_start) * 1000

        response = self._build_response(
            spatial_detections=spatial_detections,
            summary=summary,
            suggested_direction=suggested_direction,
            image_width=processed.processed_width,
            image_height=processed.processed_height,
            processing_time_ms=round(total_ms, 2),
        )

        logger.info(
            f"Analysis complete | objects: {len(detections)} | "
            f"total time: {total_ms:.1f} ms | summary: '{summary}'"
        )

        return response

    def _build_response(
        self,
        spatial_detections: List[SpatialDetection],
        summary: str,
        suggested_direction,
        image_width: int,
        image_height: int,
        processing_time_ms: float,
    ) -> AnalysisResponse:
        """
        Converts internal SpatialDetection objects into the Pydantic response schema.

        Args:
            spatial_detections: Detections annotated with direction/proximity/hazard.
            summary: Natural-language scene description.
            image_width: Width of the processed image.
            image_height: Height of the processed image.
            processing_time_ms: Total pipeline duration.

        Returns:
            Fully populated AnalysisResponse.
        """
        objects = [
            DetectedObject(
                id=idx + 1,
                label=sd.detection.label,
                confidence=sd.detection.confidence,
                bbox=BoundingBox(
                    x=sd.detection.x,
                    y=sd.detection.y,
                    width=sd.detection.width,
                    height=sd.detection.height,
                ),
                center=Center(
                    x=sd.detection.center_x,
                    y=sd.detection.center_y,
                ),
                direction=sd.direction,
                proximity=sd.proximity,
                is_hazard=sd.is_hazard,
            )
            for idx, sd in enumerate(spatial_detections)
        ]

        return AnalysisResponse(
            success=True,
            processing_time_ms=processing_time_ms,
            model_used="YOLOv8",
            image_width=image_width,
            image_height=image_height,
            objects=objects,
            object_count=len(objects),
            summary=summary,
            hazard_detected=any(sd.is_hazard for sd in spatial_detections),
            suggested_direction=suggested_direction,
        )
