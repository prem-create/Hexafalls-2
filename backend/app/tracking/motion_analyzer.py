"""
Walking Eye - AI Perception Engine
Motion Analyzer.

Transforms a TemporalBuffer of Observations into a MotionResult:

  state       — APPROACHING | MOVING_AWAY | STATIONARY | UNKNOWN
  direction   — LEFT | RIGHT | CENTER  (horizontal in image)
  confidence  — 0..1 score for the classification
  distance    — metric metres (if depth available) or None
  velocity    — metric m/s (if depth available) or relative scale-change rate

Algorithm
---------
1. Require at least MIN_TRACK_HISTORY observations.
2. If metric depth is available → use depth delta as the primary signal.
3. Otherwise  → use bbox-area relative change as a proxy.
4. Temporal smoothing: use the *mean* of consecutive pair-wise deltas
   (not just first-to-last) so transient noise from a single bad frame
   does not flip the classification.
5. Confidence is computed from:
     - number of observations (more = higher baseline)
     - consistency of pairwise deltas (low variance = higher score)
     - depth reliability flag

Important design constraint
----------------------------
We NEVER report fabricated metric distance or velocity.
If depth is unavailable:
  - distance.value   = None
  - velocity.value   = None
  - velocity.type    = "relative"
  - velocity.unit    = None
  A `relative_approach_speed` (bbox area / sec) is provided instead.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from app.tracking.temporal_buffer import Observation, TemporalBuffer
from app.utilities.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MotionState(str, Enum):
    APPROACHING  = "APPROACHING"
    MOVING_AWAY  = "MOVING_AWAY"
    STATIONARY   = "STATIONARY"
    UNKNOWN      = "UNKNOWN"


class MotionDirection(str, Enum):
    LEFT   = "LEFT"
    RIGHT  = "RIGHT"
    CENTER = "CENTER"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DistanceInfo:
    """Distance information for a tracked object."""

    value: Optional[float]    # metres, or None when unavailable
    unit: Optional[str]       # "meters" or None
    source: str               # "depth" | "relative_bbox_scale"


@dataclass
class VelocityInfo:
    """
    Velocity information.

    For metric velocity (depth available): value in m/s; type="metric".
    For relative motion (no depth): value=None, type="relative",
      relative_approach_speed holds the bbox-area/sec proxy value.
    """

    value: Optional[float]    # m/s when metric, None otherwise
    unit: Optional[str]       # "m/s" | None
    type: str                 # "metric" | "relative"
    relative_approach_speed: Optional[float] = None   # bbox-area per second


@dataclass
class MotionResult:
    """Full motion-analysis result for one tracked object."""

    state: MotionState
    direction: MotionDirection
    confidence: float          # 0.0 – 1.0

    distance: DistanceInfo
    velocity: VelocityInfo

    # Number of observations used to produce this result
    observations_used: int


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class MotionAnalyzer:
    """
    Stateless motion analyser.

    Takes a TemporalBuffer (already populated with Observations) and returns
    a MotionResult.  All thresholds are constructor-injected so they can be
    driven from Settings without hardcoding.
    """

    def __init__(
        self,
        *,
        min_track_history: int = 2,
        stationary_depth_threshold: float = 0.08,    # metres
        approaching_depth_threshold: float = 0.08,   # metres (same boundary)
        stationary_scale_threshold: float = 0.05,    # fractional area change
        approaching_scale_threshold: float = 0.05,
        direction_noise_threshold_px: int = 5,
    ) -> None:
        self.min_track_history = min_track_history
        self.stationary_depth_threshold = stationary_depth_threshold
        self.approaching_depth_threshold = approaching_depth_threshold
        self.stationary_scale_threshold = stationary_scale_threshold
        self.approaching_scale_threshold = approaching_scale_threshold
        self.direction_noise_threshold_px = direction_noise_threshold_px

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, buffer: TemporalBuffer) -> MotionResult:
        """
        Produce a MotionResult from the observations stored in `buffer`.

        Returns UNKNOWN with zero confidence when there are too few observations.
        """
        observations = buffer.as_list()
        n = len(observations)

        if n < self.min_track_history:
            return self._unknown_result(n)

        # Choose analysis path based on depth availability
        if self._has_depth(observations):
            return self._analyze_with_depth(observations)
        else:
            return self._analyze_with_bbox(observations)

    # ------------------------------------------------------------------
    # Depth-based analysis (primary, metric-accurate)
    # ------------------------------------------------------------------

    def _analyze_with_depth(self, observations: List[Observation]) -> MotionResult:
        """Use depth deltas as the ground-truth motion signal."""
        depths = [obs.estimated_depth for obs in observations]

        # Compute pairwise deltas: positive = object moved away, negative = approached
        deltas, time_deltas = self._pairwise_deltas_depth(observations)

        if not deltas:
            return self._unknown_result(len(observations))

        mean_delta = statistics.mean(deltas)           # metres per interval
        total_time = sum(time_deltas) if time_deltas else 0.0

        state = self._classify_depth_state(mean_delta)
        confidence = self._compute_depth_confidence(deltas, len(observations))
        direction = self._classify_direction(observations)

        # Metric velocity: mean depth change / mean time interval
        velocity_value: Optional[float] = None
        if total_time > 1e-6:
            total_depth_change = sum(deltas)
            raw_velocity = total_depth_change / total_time   # m/s; positive = moving away
            # Radial velocity towards camera is positive "approach speed"
            velocity_value = round(abs(raw_velocity), 3)

        latest_depth = depths[-1] if depths[-1] is not None else None

        return MotionResult(
            state=state,
            direction=direction,
            confidence=round(confidence, 3),
            distance=DistanceInfo(
                value=round(latest_depth, 2) if latest_depth is not None else None,
                unit="meters" if latest_depth is not None else None,
                source="depth",
            ),
            velocity=VelocityInfo(
                value=velocity_value,
                unit="m/s" if velocity_value is not None else None,
                type="metric",
            ),
            observations_used=len(observations),
        )

    def _pairwise_deltas_depth(
        self, observations: List[Observation]
    ) -> Tuple[List[float], List[float]]:
        """
        Returns (depth_deltas, time_deltas) for consecutive observation pairs.
        Pairs where depth is missing in either observation are skipped.
        Delta is new_depth - old_depth (positive = object moved farther away).
        """
        deltas: List[float] = []
        time_deltas: List[float] = []
        for i in range(1, len(observations)):
            prev = observations[i - 1]
            curr = observations[i]
            if prev.estimated_depth is None or curr.estimated_depth is None:
                continue
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                dt = 0.001  # guard against identical timestamps
            deltas.append(curr.estimated_depth - prev.estimated_depth)
            time_deltas.append(dt)
        return deltas, time_deltas

    def _classify_depth_state(self, mean_delta: float) -> MotionState:
        """
        Classify based on mean pairwise depth delta (metres).
          mean_delta < 0  → depth decreasing → APPROACHING
          mean_delta > 0  → depth increasing → MOVING_AWAY
          |mean_delta| within threshold → STATIONARY
        """
        if abs(mean_delta) <= self.stationary_depth_threshold:
            return MotionState.STATIONARY
        if mean_delta < 0:
            return MotionState.APPROACHING
        return MotionState.MOVING_AWAY

    def _compute_depth_confidence(
        self, deltas: List[float], n_obs: int
    ) -> float:
        """
        Confidence based on:
          - how many observations were used (more = higher baseline)
          - consistency of deltas (low coefficient-of-variation = higher score)
        """
        # Observation count factor (saturates at 1.0 for 5+ observations)
        obs_factor = min(1.0, (n_obs - 1) / 4.0)

        if len(deltas) < 2:
            consistency = 0.5
        else:
            try:
                mean = statistics.mean(deltas)
                stdev = statistics.stdev(deltas)
                cv = abs(stdev / mean) if abs(mean) > 1e-9 else (stdev * 10)
                # CV of 0 = perfectly consistent = 1.0; CV of 2 = 0.0
                consistency = max(0.0, 1.0 - cv / 2.0)
            except statistics.StatisticsError:
                consistency = 0.5

        # Depth-based analysis is inherently more reliable → add a small bonus
        depth_bonus = 0.15
        raw = obs_factor * 0.5 + consistency * 0.35 + depth_bonus
        return min(1.0, raw)

    # ------------------------------------------------------------------
    # BBox-scale fallback analysis (no depth)
    # ------------------------------------------------------------------

    def _analyze_with_bbox(self, observations: List[Observation]) -> MotionResult:
        """
        Use relative bounding-box area change as a PROXY for approach/retreat.

        We are explicit that this is relative, not metric.
        """
        areas = [obs.bbox_area for obs in observations]
        deltas, time_deltas = self._pairwise_deltas_area(observations)

        if not deltas or areas[0] == 0:
            return self._unknown_result(len(observations))

        # Normalise area delta as fraction of the first (reference) area
        ref_area = max(areas[0], 1)
        normalised_deltas = [d / ref_area for d in deltas]
        mean_norm_delta = statistics.mean(normalised_deltas)

        state = self._classify_scale_state(mean_norm_delta)
        confidence = self._compute_scale_confidence(normalised_deltas, len(observations))
        direction = self._classify_direction(observations)

        # Relative approach speed = mean bbox area change per second
        relative_approach_speed: Optional[float] = None
        total_time = sum(time_deltas) if time_deltas else 0.0
        if total_time > 1e-6 and deltas:
            relative_approach_speed = round(sum(deltas) / total_time, 1)

        return MotionResult(
            state=state,
            direction=direction,
            confidence=round(confidence, 3),
            distance=DistanceInfo(
                value=None,
                unit=None,
                source="relative_bbox_scale",
            ),
            velocity=VelocityInfo(
                value=None,
                unit=None,
                type="relative",
                relative_approach_speed=relative_approach_speed,
            ),
            observations_used=len(observations),
        )

    def _pairwise_deltas_area(
        self, observations: List[Observation]
    ) -> Tuple[List[float], List[float]]:
        """
        Returns (area_deltas, time_deltas) for consecutive pairs.
        Delta = current_area - previous_area (positive = getting bigger = approaching).
        """
        deltas: List[float] = []
        time_deltas: List[float] = []
        for i in range(1, len(observations)):
            prev = observations[i - 1]
            curr = observations[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                dt = 0.001
            deltas.append(float(curr.bbox_area - prev.bbox_area))
            time_deltas.append(dt)
        return deltas, time_deltas

    def _classify_scale_state(self, mean_norm_delta: float) -> MotionState:
        """
        Classify based on normalised bbox-area change.
          mean > threshold  → APPROACHING  (object growing = getting closer)
          mean < -threshold → MOVING_AWAY
          otherwise         → STATIONARY
        """
        if mean_norm_delta > self.approaching_scale_threshold:
            return MotionState.APPROACHING
        if mean_norm_delta < -self.approaching_scale_threshold:
            return MotionState.MOVING_AWAY
        return MotionState.STATIONARY

    def _compute_scale_confidence(
        self, normalised_deltas: List[float], n_obs: int
    ) -> float:
        """
        Confidence for bbox-scale based classification.
        Lower baseline than depth-based (inherently less reliable).
        """
        obs_factor = min(1.0, (n_obs - 1) / 4.0)

        if len(normalised_deltas) < 2:
            consistency = 0.4
        else:
            try:
                mean = statistics.mean(normalised_deltas)
                stdev = statistics.stdev(normalised_deltas)
                cv = abs(stdev / mean) if abs(mean) > 1e-9 else (stdev * 10)
                consistency = max(0.0, 1.0 - cv / 2.0)
            except statistics.StatisticsError:
                consistency = 0.4

        raw = obs_factor * 0.45 + consistency * 0.35
        return min(0.90, raw)   # cap at 0.90 — bbox proxy is never fully confident

    # ------------------------------------------------------------------
    # Direction classification
    # ------------------------------------------------------------------

    def _classify_direction(self, observations: List[Observation]) -> MotionDirection:
        """
        Classify horizontal movement using centre-X displacement between the
        oldest and latest observation.  Tiny displacements below the noise
        threshold are treated as CENTER.
        """
        if len(observations) < 2:
            return MotionDirection.CENTER

        oldest = observations[0]
        latest = observations[-1]
        dx = latest.center_x - oldest.center_x

        if abs(dx) < self.direction_noise_threshold_px:
            return MotionDirection.CENTER
        return MotionDirection.LEFT if dx < 0 else MotionDirection.RIGHT

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_depth(observations: List[Observation]) -> bool:
        """Returns True if at least 2 observations contain non-None depth."""
        count = sum(1 for o in observations if o.estimated_depth is not None)
        return count >= 2

    @staticmethod
    def _unknown_result(n_obs: int) -> MotionResult:
        """Returns a zeroed-out UNKNOWN result when there is insufficient history."""
        return MotionResult(
            state=MotionState.UNKNOWN,
            direction=MotionDirection.CENTER,
            confidence=0.0,
            distance=DistanceInfo(value=None, unit=None, source="relative_bbox_scale"),
            velocity=VelocityInfo(value=None, unit=None, type="relative"),
            observations_used=n_obs,
        )
