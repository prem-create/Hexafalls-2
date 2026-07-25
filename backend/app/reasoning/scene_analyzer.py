"""
Walking Eye - AI Perception Engine
Scene Analyzer.

Produces natural-language scene summaries with:
  - Hazard warnings (closest + most-central object first)
  - Metric distance when MiDaS depth is available
  - Motion state riders ("approaching", "moving away")
  - Suggested direction to move

Priority order in speech:
  1. Closest hazard in the walking path (with distance + motion)
  2. Scene description (remaining objects, closest first)

Examples:
  "Stop, person ahead, 2.3 metres, approaching! Move left."
  "Caution, chair ahead, 4.1 metres — stationary."
  "There is a bottle on your right."
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

from app.reasoning.spatial_analyzer import SpatialAnalyzer, SpatialDetection
from app.utilities.logger import get_logger

logger = get_logger(__name__)

_DIRECTION_PHRASES = {
    "left":         "on your left",
    "center-left":  "ahead, slightly to your left",
    "center":       "ahead",
    "center-right": "ahead, slightly to your right",
    "right":        "on your right",
}

_MOTION_PHRASES = {
    "APPROACHING": "— coming!",
    "MOVING_AWAY": "— going away.",
    "STATIONARY":  "— stationary.",
}


class SceneAnalyzer:
    """
    Converts a prioritised list of SpatialDetection objects into a
    human-readable scene summary.

    Stateless — safe to reuse across requests with no shared state.
    """

    def __init__(self) -> None:
        self._spatial = SpatialAnalyzer()

    def summarize(
        self,
        spatial_detections: List[SpatialDetection],
        frame_width: int,
        frame_height: int,
        motion_results: Optional[Dict] = None,
        tracked_detections: Optional[List] = None,
        det_to_depth: Optional[Dict] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Generates a natural-language summary.

        Args:
            spatial_detections: Already-sorted (priority order) detections.
            frame_width/height:  Processed image dimensions.
            motion_results:      track_id → MotionResult (optional).
            tracked_detections:  TrackedDetection list for id→track_id lookup.
            det_to_depth:        detection python-id → depth_m (optional).

        Returns:
            (summary_text, suggested_direction)
        """
        if not spatial_detections:
            return "I couldn't identify any object.", None

        # Build detection python-id → track_id lookup
        det_to_track: Dict[int, int] = {}
        if tracked_detections:
            for td in tracked_detections:
                det_to_track[id(td.detection)] = td.track_id

        # spatial_detections already sorted by priority from SpatialAnalyzer
        ordered = spatial_detections

        suggested_direction = self._spatial.suggest_clear_direction(
            spatial_detections, frame_width, frame_height
        )

        warning = self._build_hazard_warning(
            ordered,
            suggested_direction,
            motion_results=motion_results or {},
            det_to_track=det_to_track,
            det_to_depth=det_to_depth or {},
        )
        description = self._build_scene_description(
            ordered,
            motion_results=motion_results or {},
            det_to_track=det_to_track,
            det_to_depth=det_to_depth or {},
        )

        summary = f"{warning} {description}".strip() if warning else description
        if not warning and suggested_direction:
            summary = f"{summary} Go to {suggested_direction}.".strip()
        logger.debug(f"Scene summary: '{summary}'")
        return summary, suggested_direction

    # ------------------------------------------------------------------
    # Hazard warning
    # ------------------------------------------------------------------

    def _build_hazard_warning(
        self,
        ordered: List[SpatialDetection],
        suggested_direction: Optional[str],
        motion_results: Dict,
        det_to_track: Dict[int, int],
        det_to_depth: Dict[int, Optional[float]],
    ) -> str:
        hazards = [sd for sd in ordered if sd.is_hazard]
        if not hazards:
            return ""

        closest = hazards[0]
        label           = closest.detection.label
        direction_phrase = _DIRECTION_PHRASES[closest.direction]
        move_phrase     = f" Go to {suggested_direction}." if suggested_direction else ""

        # Distance phrase — prefer MiDaS depth, fall back to proximity label
        depth_m = det_to_depth.get(id(closest.detection)) or closest.depth_m
        distance_phrase = self._distance_phrase(depth_m, closest.proximity)

        # Motion rider
        motion_rider = ""
        track_id = det_to_track.get(id(closest.detection))
        if track_id is not None and track_id in motion_results:
            mr = motion_results[track_id]
            state_str = mr.state.value if hasattr(mr.state, "value") else str(mr.state)
            if mr.confidence >= 0.40:
                motion_rider = " " + _MOTION_PHRASES.get(state_str, "")

        if closest.proximity == "very close":
            return (
                f"Stop, {label} {direction_phrase}"
                f"{distance_phrase}, very close!"
                f"{motion_rider}{move_phrase}"
            )
        return (
            f"Caution, {label} {direction_phrase}"
            f"{distance_phrase}."
            f"{motion_rider}{move_phrase}"
        )

    def _distance_phrase(
        self,
        depth_m: Optional[float],
        proximity: str,
    ) -> str:
        """
        Returns a distance string to embed in the warning, e.g. ', 3.2 metres'.
        Uses metric depth when available; falls back to proximity label.
        """
        if depth_m is not None:
            return f", {depth_m:.1f} meters"
        # Proximity-label fallback (no number, but gives qualitative info)
        if proximity in ("very close", "close"):
            return ""   # already covered by "very close!" / "close" in caller
        return ""

    # ------------------------------------------------------------------
    # Scene description
    # ------------------------------------------------------------------

    def _build_scene_description(
        self,
        ordered: List[SpatialDetection],
        motion_results: Dict,
        det_to_track: Dict[int, int],
        det_to_depth: Dict[int, Optional[float]],
    ) -> str:
        if not ordered:
            return ""

        # Focus on near/medium objects (proximity != "far")
        near_ordered = [sd for sd in ordered if sd.proximity != "far"]
        if not near_ordered:
            return ""

        parts = []
        for sd in near_ordered:
            parts.append(
                self._describe_detection(
                    sd,
                    motion_results=motion_results,
                    det_to_track=det_to_track,
                    det_to_depth=det_to_depth,
                )
            )

        joined = self._join_with_oxford_comma(parts)
        return f"There is {joined}."

    def _describe_detection(
        self,
        sd: SpatialDetection,
        motion_results: Dict,
        det_to_track: Dict[int, int],
        det_to_depth: Dict[int, Optional[float]],
    ) -> str:
        label = sd.detection.label
        direction_phrase = _DIRECTION_PHRASES[sd.direction]

        # Distance (prefer det_to_depth or sd.depth_m)
        depth_m = det_to_depth.get(id(sd.detection)) or sd.depth_m
        distance_phrase = ""
        if depth_m is not None:
            distance_phrase = f", {depth_m:.1f} meters away"

        # Motion status
        motion_phrase = ""
        track_id = det_to_track.get(id(sd.detection))
        if track_id is not None and track_id in motion_results:
            mr = motion_results[track_id]
            state_str = mr.state.value if hasattr(mr.state, "value") else str(mr.state)
            if mr.confidence >= 0.40:
                if state_str == "APPROACHING":
                    motion_phrase = ", coming"
                elif state_str == "MOVING_AWAY":
                    motion_phrase = ", going away"
                elif state_str == "STATIONARY":
                    motion_phrase = ", stationary"

        return f"a {label} {direction_phrase}{distance_phrase}{motion_phrase}"

    def _join_with_oxford_comma(self, items: List[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _pluralize(self, word: str) -> str:
        irregulars = {
            "person": "people", "mouse": "mice",
            "knife": "knives", "leaf": "leaves",
        }
        if word in irregulars:
            return irregulars[word]
        if word.endswith(("s", "x", "z")):
            return word + "es"
        if word.endswith(("ch", "sh")):
            return word + "es"
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        return word + "s"
