"""
Walking Eye - AI Perception Engine
Spatial Analyzer.

Adds directional and proximity awareness to raw detections:

- Direction: where the object sits in the frame
  (left / center-left / center / center-right / right)
- Proximity: how close the object appears to be.
  When metric depth is available (from MiDaS), proximity buckets are
  derived from actual metres.  Without depth, bbox area ratio is used
  as a relative proxy.
- Hazard flag: proximity + direction → "close and in your path" signal.
- Priority sorting: objects are ranked by a combined score that weights
  both depth (closest first) AND how central they are to the walking
  path.  This ensures the TTS always leads with the most urgent object.

This module knows nothing about YOLO, OpenCV, or HTTP.  Stateless.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.utilities.logger import get_logger
from app.vision.detector import Detection

logger = get_logger(__name__)


@dataclass
class SpatialDetection:
    """A Detection enriched with direction, proximity, hazard, and depth info."""

    detection: Detection
    direction: str        # "left" | "center-left" | "center" | "center-right" | "right"
    proximity: str        # "far" | "medium" | "close" | "very close"
    is_hazard: bool
    depth_m: Optional[float] = field(default=None)   # metres from MiDaS, or None


class SpatialAnalyzer:
    """
    Converts raw Detection objects into spatially-aware SpatialDetection objects.
    """

    # ── Direction buckets (fraction of frame width) ──────────────────────
    LEFT_BOUND        = 0.35
    CENTER_LEFT_BOUND = 0.45
    CENTER_RIGHT_BOUND= 0.55
    RIGHT_BOUND       = 0.65

    # ── Proximity buckets — bbox-area fraction (used when no depth) ───────
    FAR_THRESHOLD    = 0.04
    MEDIUM_THRESHOLD = 0.15
    CLOSE_THRESHOLD  = 0.35

    # ── Proximity buckets — metric depth (metres) ─────────────────────────
    DEPTH_VERY_CLOSE_M = 1.5
    DEPTH_CLOSE_M      = 3.0
    DEPTH_MEDIUM_M     = 6.0
    # anything beyond DEPTH_MEDIUM_M → "far"

    # ── Rank used for legacy sort-by-proximity ────────────────────────────
    _PROXIMITY_RANK = {"very close": 0, "close": 1, "medium": 2, "far": 3}

    # ── Centre-path priority weight ───────────────────────────────────────
    # Objects in the direct walking path score higher than off-centre ones.
    _DIRECTION_CENTRE_WEIGHT = {
        "center":       1.0,
        "center-left":  0.7,
        "center-right": 0.7,
        "left":         0.3,
        "right":        0.3,
    }

    # ── Proximity occupancy weights (for clearance suggestion) ────────────
    _PROXIMITY_WEIGHT = {"very close": 1.0, "close": 0.6, "medium": 0.25, "far": 0.05}
    _MIN_CLEARANCE_MARGIN = 0.15

    # ── Maximum depth considered "in range" for priority scoring ──────────
    _MAX_DEPTH_FOR_PRIORITY = 20.0   # metres

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        depths: Optional[List[Optional[float]]] = None,
    ) -> List[SpatialDetection]:
        """
        Annotates each detection with direction, proximity, hazard status,
        and depth.  Returned list is sorted by combined priority score
        (closest + most-central first) so callers always get the most
        important object at index 0.

        Args:
            detections:  Raw YOLO detections.
            frame_width: Processed image width (px).
            frame_height: Processed image height (px).
            depths:      Optional per-detection depths in metres, parallel
                         to detections.  None entries fall back to bbox proxy.
        """
        frame_area = max(frame_width * frame_height, 1)
        results: List[SpatialDetection] = []

        for i, det in enumerate(detections):
            depth_m: Optional[float] = None
            if depths is not None and i < len(depths):
                depth_m = depths[i]

            direction = self._classify_direction(det.center_x, frame_width)
            proximity = self._classify_proximity(det, frame_area, depth_m)
            is_hazard = (
                proximity in ("close", "very close")
                and direction in ("center-left", "center", "center-right")
            )

            results.append(SpatialDetection(
                detection=det,
                direction=direction,
                proximity=proximity,
                is_hazard=is_hazard,
                depth_m=depth_m,
            ))

        if any(r.is_hazard for r in results):
            logger.info("Hazard detected: object close and in the walking path.")

        # Sort by priority: closest + most-central first
        results.sort(key=lambda sd: self._priority_score(sd), reverse=True)
        return results

    def sort_by_proximity(
        self, spatial_detections: List[SpatialDetection]
    ) -> List[SpatialDetection]:
        """
        Returns a new list sorted by combined depth+centrality priority.
        Falls back to proximity-rank sort when no depth is available.
        """
        has_depth = any(sd.depth_m is not None for sd in spatial_detections)
        if has_depth:
            return sorted(
                spatial_detections,
                key=lambda sd: self._priority_score(sd),
                reverse=True,
            )
        return sorted(
            spatial_detections,
            key=lambda sd: self._PROXIMITY_RANK.get(sd.proximity, 99),
        )

    def suggest_clear_direction(
        self,
        spatial_detections: List[SpatialDetection],
        frame_width: int,
        frame_height: int,
    ) -> Optional[str]:
        """
        Suggests 'left' or 'right' when a hazard is blocking the path and
        one side is meaningfully clearer than the other.
        """
        if not any(sd.proximity in ("close", "very close") for sd in spatial_detections):
            return None

        frame_area = max(frame_width * frame_height, 1)
        frame_mid  = frame_width / 2

        left_score  = 0.0
        right_score = 0.0

        for sd in spatial_detections:
            det    = sd.detection
            weight = self._PROXIMITY_WEIGHT.get(sd.proximity, 0.1)

            # If we have metric depth, weight more strongly by distance
            if sd.depth_m is not None:
                # closer objects block more — inverse relationship
                depth_weight = 1.0 / max(sd.depth_m, 0.5)
                weight = weight * (1.0 + depth_weight)

            area_ratio = (det.width * det.height) / frame_area
            occupancy  = area_ratio * weight

            if det.center_x < frame_mid:
                left_score  += occupancy
            else:
                right_score += occupancy

        total = left_score + right_score
        if total <= 0:
            return None

        margin = abs(left_score - right_score) / total
        if margin < self._MIN_CLEARANCE_MARGIN:
            return None

        return "left" if left_score < right_score else "right"

    # ------------------------------------------------------------------
    # Priority scoring
    # ------------------------------------------------------------------

    def _priority_score(self, sd: SpatialDetection) -> float:
        """
        Higher score = more urgent = announced first.

        Score = depth_score * centre_weight

        depth_score:
          - When metric depth available: 1 - (depth_m / MAX_DEPTH)
            → 1.0 at 0 m, 0.0 at MAX_DEPTH m
          - When no depth: use bbox-area fraction as proxy (0–1)

        centre_weight: direction-based multiplier (1.0 for center, 0.3 for sides)
        """
        centre_weight = self._DIRECTION_CENTRE_WEIGHT.get(sd.direction, 0.5)

        if sd.depth_m is not None:
            depth_score = max(
                0.0,
                1.0 - sd.depth_m / self._MAX_DEPTH_FOR_PRIORITY
            )
        else:
            # bbox area fraction as proxy
            det = sd.detection
            area = det.width * det.height
            depth_score = min(1.0, area / 100_000.0)   # normalised

        return depth_score * centre_weight

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

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

    def _classify_proximity(
        self,
        det: Detection,
        frame_area: int,
        depth_m: Optional[float],
    ) -> str:
        """
        Uses metric depth when available; falls back to bbox-area ratio.
        """
        if depth_m is not None:
            if depth_m <= self.DEPTH_VERY_CLOSE_M:
                return "very close"
            if depth_m <= self.DEPTH_CLOSE_M:
                return "close"
            if depth_m <= self.DEPTH_MEDIUM_M:
                return "medium"
            return "far"

        # bbox-area proxy
        bbox_area = det.width * det.height
        ratio = bbox_area / frame_area
        if ratio < self.FAR_THRESHOLD:
            return "far"
        if ratio < self.MEDIUM_THRESHOLD:
            return "medium"
        if ratio < self.CLOSE_THRESHOLD:
            return "close"
        return "very close"
