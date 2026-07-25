"""
Walking Eye - AI Perception Engine
Scene Analyzer.

Takes a list of spatially-aware detections and produces a
natural-language summary of the scene, including directional
guidance ("on your left") and proximity warnings ("very close ahead").

This module is deliberately isolated from YOLO, OpenCV, and HTTP.
It only knows about SpatialDetection objects and strings.

Future upgrade path:
- Swap `_build_scene_description` for an LLM call (Claude, GPT, local
  model) without changing any other module, just this file.
- Add scene classification (indoor/outdoor, crowded/empty).
"""

from collections import Counter
from typing import List, Optional, Tuple

from app.reasoning.spatial_analyzer import SpatialAnalyzer, SpatialDetection
from app.utilities.logger import get_logger

logger = get_logger(__name__)

_DIRECTION_PHRASES = {
    "left": "on your left",
    "center-left": "ahead, slightly to your left",
    "center": "ahead",
    "center-right": "ahead, slightly to your right",
    "right": "on your right",
}


class SceneAnalyzer:
    """
    Converts a list of SpatialDetection objects into a human-readable
    scene summary, prioritizing safety warnings before general description.

    Stateless, safe to reuse across requests with no shared state.
    """

    def __init__(self) -> None:
        self._spatial = SpatialAnalyzer()

    def summarize(
        self,
        spatial_detections: List[SpatialDetection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[str, Optional[str]]:
        """
        Generates a natural-language summary of detected objects.

        Returns:
            Tuple of (summary_text, suggested_direction). suggested_direction
            is 'left', 'right', or None (no hazard, or no clearly better side).
        """
        if not spatial_detections:
            logger.debug("No detections, returning empty scene message.")
            return "I couldn't identify any object.", None

        ordered = self._spatial.sort_by_proximity(spatial_detections)
        suggested_direction = self._spatial.suggest_clear_direction(
            spatial_detections, frame_width, frame_height
        )

        warning = self._build_hazard_warning(ordered, suggested_direction)
        description = self._build_scene_description(ordered)

        summary = f"{warning} {description}".strip() if warning else description
        logger.debug(f"Scene summary: '{summary}'")
        return summary, suggested_direction

    def _build_hazard_warning(
        self, ordered: List[SpatialDetection], suggested_direction: Optional[str]
    ) -> str:
        hazards = [sd for sd in ordered if sd.is_hazard]
        if not hazards:
            return ""

        closest = hazards[0]
        label = closest.detection.label
        direction_phrase = _DIRECTION_PHRASES[closest.direction]
        move_phrase = f" Move {suggested_direction}." if suggested_direction else ""

        if closest.proximity == "very close":
            return f"Stop, {label} {direction_phrase}, very close!{move_phrase}"
        return f"Caution, {label} {direction_phrase}.{move_phrase}"

    def _build_scene_description(self, ordered: List[SpatialDetection]) -> str:
        total_objects = len(ordered)

        if total_objects == 1:
            sd = ordered[0]
            return f"There is a {sd.detection.label} {_DIRECTION_PHRASES[sd.direction]}."

        groups: Counter = Counter()
        order_seen: List[Tuple[str, str]] = []
        for sd in ordered:
            key = (sd.detection.label, sd.direction)
            if key not in groups:
                order_seen.append(key)
            groups[key] += 1

        parts = [
            self._phrase_for_group(label, direction, groups[(label, direction)])
            for (label, direction) in order_seen
        ]

        # Single repeated group (e.g. "3 bottles on your left") needs "are";
        # a mixed list (e.g. "a chair ahead and a bottle on your left") reads
        # naturally with "is", matching how people actually say it aloud.
        verb = "are" if len(order_seen) == 1 else "is"

        joined = self._join_with_oxford_comma(parts)
        return f"There {verb} {joined} visible."

    def _phrase_for_group(self, label: str, direction: str, count: int) -> str:
        direction_phrase = _DIRECTION_PHRASES[direction]
        if count == 1:
            return f"a {label} {direction_phrase}"
        return f"{count} {self._pluralize(label)} {direction_phrase}"

    def _join_with_oxford_comma(self, items: List[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _pluralize(self, word: str) -> str:
        irregulars = {
            "person": "people",
            "mouse": "mice",
            "knife": "knives",
            "leaf": "leaves",
        }

        if word in irregulars:
            return irregulars[word]

        if word.endswith("s") or word.endswith("x") or word.endswith("z"):
            return word + "es"

        if word.endswith("ch") or word.endswith("sh"):
            return word + "es"

        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"

        return word + "s"
