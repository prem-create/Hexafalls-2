"""
Walking Eye - AI Perception Engine
Analysis Service.

Orchestrates the full perception pipeline:
  raw bytes
  → image processing
  → object detection      (YOLO)
  → depth estimation      (MiDaS DPT_Hybrid)
  → object tracking       (IoU tracker — persistent track IDs)
  → temporal history      (ring buffer per track)
  → motion analysis       (approach / velocity / direction)
  → alert management      (deduplication, zones, priority, should_speak)
  → spatial awareness     (direction, proximity, hazard)
  → scene reasoning       (natural-language summary)
  → structured response

This is the only place that knows about all layers.
"""

import time
from typing import Dict, List, Optional

from ultralytics import YOLO

from app.alerting.alert_manager import AlertEvent
from app.config.settings import get_settings
from app.reasoning.scene_analyzer import SceneAnalyzer
from app.reasoning.spatial_analyzer import SpatialAnalyzer, SpatialDetection
from app.schemas.analysis import (
    AlertInfo,
    AnalysisResponse,
    BoundingBox,
    Center,
    DetectedObject,
    DistanceInfo,
    MotionInfo,
    VelocityInfo,
)
from app.tracking.motion_analyzer import MotionResult
from app.tracking.tracker import TrackedDetection
from app.tracking.tracker_store import TrackerStore
from app.utilities.logger import get_logger
from app.vision.depth_estimator import DepthEstimator
from app.vision.detector import Detection, DetectionError, ObjectDetector
from app.vision.image_processor import ImageProcessingError, ImageProcessor

logger = get_logger(__name__)
settings = get_settings()


class AnalysisService:
    """
    Orchestrates image analysis from raw bytes to structured response.
    All dependencies are injected — fully testable without a running app.
    """

    def __init__(
        self,
        model: YOLO,
        tracker_store: Optional[TrackerStore] = None,
        depth_estimator: Optional[DepthEstimator] = None,
    ) -> None:
        self._processor = ImageProcessor(max_dimension=settings.MAX_IMAGE_DIMENSION)
        self._detector = ObjectDetector(
            model=model,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        )
        self._spatial = SpatialAnalyzer()
        self._analyzer = SceneAnalyzer()
        self._tracker_store = tracker_store
        self._depth_estimator = depth_estimator

    async def analyze(
        self,
        image_bytes: bytes,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        estimated_depths: Optional[List[Optional[float]]] = None,
    ) -> AnalysisResponse:
        """
        Full pipeline: raw bytes → AnalysisResponse with alerts.

        Args:
            image_bytes:       Raw bytes from the uploaded image.
            session_id:        Client session ID for tracking continuity.
            timestamp:         Unix timestamp. Defaults to now.
            estimated_depths:  External per-detection depths (override MiDaS).
        """
        pipeline_start = time.perf_counter()
        if timestamp is None:
            timestamp = time.time()

        tracking_active = self._tracker_store is not None and session_id is not None
        depth_active = (
            self._depth_estimator is not None and self._depth_estimator.is_loaded
        )

        # ── Step 1: Image Processing ──────────────────────────────────────
        logger.info("Step 1: Processing image...")
        processed = self._processor.process(image_bytes)

        if processed.was_resized:
            logger.info(
                f"Image resized: {processed.original_width}x{processed.original_height}"
                f" → {processed.processed_width}x{processed.processed_height}"
            )

        # ── Step 2: Object Detection ──────────────────────────────────────
        logger.info("Step 2: Running object detection...")
        detections, inference_ms = self._detector.detect(processed.array)
        logger.info(f"Detection: {len(detections)} object(s) in {inference_ms:.1f} ms")

        # ── Step 3: Depth Estimation ──────────────────────────────────────
        if estimated_depths is not None:
            depths = estimated_depths
        elif depth_active and detections:
            logger.info("Step 3: Running MiDaS depth estimation...")
            try:
                depths = self._depth_estimator.estimate_depths(
                    processed.array, detections
                )
                valid = sum(1 for d in depths if d is not None)
                logger.info(f"Depth: {valid}/{len(detections)} valid")
            except Exception as e:
                logger.warning(f"Depth estimation failed (degrading): {e}")
                depths = [None] * len(detections)
        else:
            depths = [None] * len(detections)

        # ── Step 4: Tracking + Motion Analysis + Alerts ───────────────────
        tracked_detections: List[TrackedDetection] = []
        motion_results: Dict[int, MotionResult] = {}
        alert_events: List[AlertEvent] = []

        if tracking_active:
            logger.info("Step 4: Running tracking, motion analysis, and alerts...")
            try:
                tracked_detections, motion_results, alert_events = (
                    self._tracker_store.process_frame(
                        session_id=session_id,
                        detections=detections,
                        timestamp=timestamp,
                        estimated_depths=(
                            depths if any(d is not None for d in depths) else None
                        ),
                    )
                )
                speakable = sum(1 for a in alert_events if a.should_speak)
                logger.info(
                    f"Tracking: {len(tracked_detections)} tracked | "
                    f"{len(alert_events)} alerts ({speakable} speakable)"
                )
            except Exception as e:
                logger.warning(f"Tracking/alert step failed (degrading): {e}")
                tracked_detections = []
                motion_results = {}
                alert_events = []

        # Build lookups
        det_to_track: Dict[int, tuple] = {
            id(td.detection): (td.track_id, motion_results.get(td.track_id))
            for td in tracked_detections
        }
        det_to_depth: Dict[int, Optional[float]] = {
            id(det): depths[i] for i, det in enumerate(detections)
        }

        # ── Step 5: Spatial Awareness ─────────────────────────────────────
        logger.info("Step 5: Analyzing direction and proximity...")
        spatial_detections = self._spatial.analyze(
            detections,
            frame_width=processed.processed_width,
            frame_height=processed.processed_height,
            depths=depths if any(d is not None for d in depths) else None,
        )

        # ── Step 6: Scene Reasoning ───────────────────────────────────────
        logger.info("Step 6: Generating scene summary...")
        summary, suggested_direction = self._analyzer.summarize(
            spatial_detections,
            frame_width=processed.processed_width,
            frame_height=processed.processed_height,
            motion_results=motion_results if tracking_active else {},
            tracked_detections=tracked_detections,
            det_to_depth=det_to_depth,
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        response = self._build_response(
            spatial_detections=spatial_detections,
            det_to_track=det_to_track,
            det_to_depth=det_to_depth,
            alert_events=alert_events,
            summary=summary,
            suggested_direction=suggested_direction,
            image_width=processed.processed_width,
            image_height=processed.processed_height,
            processing_time_ms=round(total_ms, 2),
            session_id=session_id,
            tracking_enabled=tracking_active,
            depth_enabled=depth_active,
        )

        logger.info(
            f"Analysis complete | objects: {len(detections)} | "
            f"depth: {'on' if depth_active else 'off'} | "
            f"alerts: {len(alert_events)} | total: {total_ms:.1f} ms"
        )
        return response

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    def _build_response(
        self,
        spatial_detections: List[SpatialDetection],
        det_to_track: Dict[int, tuple],
        det_to_depth: Dict[int, Optional[float]],
        alert_events: List[AlertEvent],
        summary: str,
        suggested_direction,
        image_width: int,
        image_height: int,
        processing_time_ms: float,
        session_id: Optional[str],
        tracking_enabled: bool,
        depth_enabled: bool,
    ) -> AnalysisResponse:

        objects = []
        for idx, sd in enumerate(spatial_detections):
            det = sd.detection
            track_id, motion_result = det_to_track.get(id(det), (None, None))
            depth_m = det_to_depth.get(id(det))

            motion_info: Optional[MotionInfo] = None
            if motion_result is not None:
                dist_value  = depth_m if depth_m is not None else motion_result.distance.value
                dist_unit   = "meters" if depth_m is not None else motion_result.distance.unit
                dist_source = "depth"  if depth_m is not None else motion_result.distance.source

                motion_info = MotionInfo(
                    state=motion_result.state.value,
                    direction=motion_result.direction.value,
                    confidence=motion_result.confidence,
                    observations_used=motion_result.observations_used,
                    distance=DistanceInfo(
                        value=dist_value,
                        unit=dist_unit,
                        source=dist_source,
                    ),
                    velocity=VelocityInfo(
                        value=motion_result.velocity.value,
                        unit=motion_result.velocity.unit,
                        type=motion_result.velocity.type,
                        relative_approach_speed=motion_result.velocity.relative_approach_speed,
                    ),
                )
            elif depth_m is not None:
                # Depth available but not yet enough history for motion
                motion_info = MotionInfo(
                    state="UNKNOWN",
                    direction="CENTER",
                    confidence=0.0,
                    observations_used=0,
                    distance=DistanceInfo(value=depth_m, unit="meters", source="depth"),
                    velocity=VelocityInfo(value=None, unit=None, type="relative"),
                )

            objects.append(DetectedObject(
                id=idx + 1,
                label=det.label,
                confidence=det.confidence,
                bbox=BoundingBox(x=det.x, y=det.y,
                                 width=det.width, height=det.height),
                center=Center(x=det.center_x, y=det.center_y),
                direction=sd.direction,
                proximity=sd.proximity,
                is_hazard=sd.is_hazard,
                track_id=track_id,
                motion=motion_info,
            ))

        # Convert AlertEvent → AlertInfo schema objects
        alerts_schema = [
            AlertInfo(
                track_id=ae.track_id,
                label=ae.label,
                alert_type=ae.alert_type.value,
                priority=ae.priority.value,
                message=ae.message,
                should_speak=ae.should_speak,
                distance_m=ae.distance_m,
                zone=ae.zone.value if ae.zone else None,
                motion_state=ae.motion_state,
                velocity_ms=ae.velocity_ms,
            )
            for ae in alert_events
        ]

        return AnalysisResponse(
            success=True,
            processing_time_ms=processing_time_ms,
            model_used="YOLO11n + MiDaS DPT_Hybrid" if depth_enabled else "YOLO11n",
            image_width=image_width,
            image_height=image_height,
            objects=objects,
            object_count=len(objects),
            summary=summary,
            hazard_detected=any(sd.is_hazard for sd in spatial_detections),
            suggested_direction=suggested_direction,
            session_id=session_id,
            tracking_enabled=tracking_enabled,
            alerts=alerts_schema,
        )
