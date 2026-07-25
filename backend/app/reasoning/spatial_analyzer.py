"""
Walking Eye - AI Perception Engine
Spatial Analyzer.

Adds directional and proximity awareness to raw detections:

- Direction: where the object sits in the frame
  (left / center-left / center / center-right / right)
- Proximity: how close the object appears to be, estimated from how
  much of the frame its bounding box fills. This is a cheap, model-free
  stand-in for real depth estimation — bigger box (relative to the
  frame) = closer object.
- Hazard flag: proximity + direction combine into a simple "close and
  in your path" warning signal, which the scene analyzer surfaces first.

This module knows nothing about YOLO, OpenCV, or HTTP. It only
transforms Detection objects using bbox/center fields that already
exist, plus the frame dimensions. Stateless — safe to reuse across
requests.
"""

from dataclasses import dataclass
from typing import List, Optional

from app.utilities.logger import get_logger
from app.vision.detector import Detection

logger = get_logger(__name__)


@dataclass
class SpatialDetection:
    """A Detection enriched with direction, proximity, and hazard info."""

    detection: Detection
    direction: str    # "left", "center-left", "center", "center-right", "right"
    proximity: str    # "far", "medium", "close", "very close"
    is_hazard: bool   # True if close/very-close AND roughly in the walking path


class SpatialAnalyzer:
    """
    Converts raw Detection objects into spatially-aware SpatialDetection
    objects using simple geometry — no ML required, no extra latency.
    """

    # Direction buckets, as a proportion of frame width (0.0 = left edge)
    LEFT_BOUND = 0.35
    CENTER_LEFT_BOUND = 0.45
    CENTER_RIGHT_BOUND = 0.55
    RIGHT_BOUND = 0.65

    # Proximity buckets, as a proportion of frame area covered by the bbox
    FAR_THRESHOLD = 0.04
    MEDIUM_THRESHOLD = 0.15
    CLOSE_THRESHOLD = 0.35

    # Order used to rank proximity from closest to farthest
    _PROXIMITY_RANK = {"very close": 0, "close": 1, "medium": 2, "far": 3}

    def analyze(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> List[SpatialDetection]:
        """
        Annotates each detection with direction, proximity, and hazard status.

        Args:
            detections: Raw detections from the object detector.
            frame_width: Width of the processed image (pixels).
            frame_height: Height of the processed image (pixels).

        Returns:
            List of SpatialDetection, same order as input, sorted by
            nothing in particular — callers sort as needed.
        """
        frame_area = max(frame_width * frame_height, 1)
        results: List[SpatialDetection] = []

        for det in detections:
            direction = self._classify_direction(det.center_x, frame_width)
            proximity = self._classify_proximity(det, frame_area)
            is_hazard = proximity in ("close", "very close") and direction in (
                "center-left",
                "center",
                "center-right",
            )

            results.append(
                SpatialDetection(
                    detection=det,
                    direction=direction,
                    proximity=proximity,
                    is_hazard=is_hazard,
                )
            )

        if any(r.is_hazard for r in results):
            logger.info("Hazard detected: object close and in the walking path.")

        return results

    def sort_by_proximity(
        self, spatial_detections: List[SpatialDetection]
    ) -> List[SpatialDetection]:
        """Returns a new list sorted closest-first (useful for prioritizing warnings)."""
        return sorted(
            spatial_detections,
            key=lambda sd: self._PROXIMITY_RANK.get(sd.proximity, 99),
        )

    # Proximity contributes more "occupancy weight" the closer an object is,
    # so a very-close object in your path outweighs a far one at the same spot.
    _PROXIMITY_WEIGHT = {"very close": 1.0, "close": 0.6, "medium": 0.25, "far": 0.05}

    # How much clearer one side must be than the other before we bother
    # suggesting it — avoids flip-flopping between near-identical scenes.
    _MIN_CLEARANCE_MARGIN = 0.15

    def suggest_clear_direction(
        self,
        spatial_detections: List[SpatialDetection],
        frame_width: int,
        frame_height: int,
    ) -> Optional[str]:
        """
        When a hazard is blocking the path, suggests which side ('left'
        or 'right') has meaningfully more open space to step toward.

        This is a heuristic, not real free-space mapping: it sums up how
        much of the frame is "occupied" on each side (bbox area, weighted
        by how close each object is) and picks the side with less of it.

        Returns:
            'left' or 'right' if one side is meaningfully clearer, or
            None if there's no hazard, or neither side is clearly better
            (e.g. both blocked, or it's a coin flip).
        """
        if not any(sd.is_hazard for sd in spatial_detections):
            return None

        frame_area = max(frame_width * frame_height, 1)
        frame_mid = frame_width / 2

        left_score = 0.0
        right_score = 0.0

        for sd in spatial_detections:
            det = sd.detection
            weight = self._PROXIMITY_WEIGHT.get(sd.proximity, 0.1)
            area_ratio = (det.width * det.height) / frame_area
            occupancy = area_ratio * weight

            if det.center_x < frame_mid:
                left_score += occupancy
            else:
                right_score += occupancy

        total = left_score + right_score
        if total <= 0:
            return None

        margin = abs(left_score - right_score) / total
        if margin < self._MIN_CLEARANCE_MARGIN:
            return None

        return "left" if left_score < right_score else "right"

    def _classify_direction(self, center_x: int, frame_width: int) -> str:
        if frame_width <= 0:
            return "center"

        ratio = center_x / frame_width

        if ratio < self.LEFT_BOUND:
            return "left"
        if ratio < self.CENTER_LEFT_BOUND:
            return "center-left"
        if ratio <= self.CENTER_RIGHT_BOUND:
            return "center"
        if ratio <= self.RIGHT_BOUND:
            return "center-right"
        return "right"

    def _classify_proximity(self, det: Detection, frame_area: int) -> str:
        bbox_area = det.width * det.height
        ratio = bbox_area / frame_area

        if ratio < self.FAR_THRESHOLD:
            return "far"
        if ratio < self.MEDIUM_THRESHOLD:
            return "medium"
        if ratio < self.CLOSE_THRESHOLD:
            return "close"
        return "very close"
