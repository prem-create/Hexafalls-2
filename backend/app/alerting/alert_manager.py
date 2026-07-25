"""
Walking Eye - AI Perception Engine
Alert Manager.

Decides WHEN and WHAT to tell the user about detected objects.

The core problem it solves
--------------------------
The pipeline runs at ~0.67–2 FPS.  Without deduplication, every frame
would generate a speech event, causing non-stop repetitive narration:
    "Person 3 metres ahead."
    "Person 3 metres ahead."
    "Person 3 metres ahead."   ← deeply annoying, safety risk

The AlertManager maintains per-track state and only fires an alert when
something *meaningful* has changed:

    New object              → always alert
    Zone crossing           → alert  (e.g. FAR → MEDIUM)
    Motion state change     → alert  (e.g. STATIONARY → APPROACHING)
    Significant distance Δ  → alert  (configurable threshold)
    Rapid approach          → high-priority alert
    Same state, cooldown    → suppress

Architecture
------------
* One AlertManager instance per tracking session (stored in SessionState).
* Stateful: keeps TrackAlertState per track_id.
* All thresholds are constructor-injected from Settings.
* Output: List[AlertEvent] — one per object that should generate speech.
  Each AlertEvent has `should_speak: bool`, `priority`, `message`.
* The backend includes AlertEvents in the API response so the frontend
  only needs to check `should_speak=true` and speak the message verbatim.

Distance zones
--------------
FAR      > ZONE_FAR_M            (e.g. > 5 m)
MEDIUM   > ZONE_MEDIUM_M         (e.g. 3–5 m)
NEAR     > ZONE_NEAR_M           (e.g. 1.5–3 m)
VERY_NEAR ≤ ZONE_NEAR_M          (e.g. < 1.5 m)

Object priority tiers
---------------------
HIGH    — vehicle, motorcycle, bicycle, bus, truck (immediate safety risk)
MEDIUM  — person, door, stairs, escalator           (navigation relevant)
LOW     — furniture, food, background objects       (informational only)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from app.utilities.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AlertPriority(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class AlertType(str, Enum):
    NEW_OBJECT          = "NEW_OBJECT"
    ZONE_CROSSING       = "ZONE_CROSSING"
    MOTION_STATE_CHANGE = "MOTION_STATE_CHANGE"
    RAPID_APPROACH      = "RAPID_APPROACH"
    DISTANCE_UPDATE     = "DISTANCE_UPDATE"
    OBJECT_GONE         = "OBJECT_GONE"
    PERIODIC_REMINDER   = "PERIODIC_REMINDER"


class DistanceZone(str, Enum):
    VERY_NEAR = "VERY_NEAR"   # < 1.5 m
    NEAR      = "NEAR"        # 1.5–3 m
    MEDIUM    = "MEDIUM"      # 3–5 m
    FAR       = "FAR"         # > 5 m
    UNKNOWN   = "UNKNOWN"     # no depth data


# ---------------------------------------------------------------------------
# Object priority tier lookup
# ---------------------------------------------------------------------------

_HIGH_PRIORITY_LABELS = {
    "car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "bike",
    "vehicle", "van", "train", "boat",
}

_MEDIUM_PRIORITY_LABELS = {
    "person", "people", "human",
    "door", "gate", "stairs", "staircase", "escalator", "elevator",
    "fire hydrant", "stop sign", "traffic light",
}

# Everything not in HIGH or MEDIUM → LOW


def _object_priority(label: str) -> AlertPriority:
    """Returns the alert priority tier for a detected object label."""
    l = label.lower()
    if l in _HIGH_PRIORITY_LABELS:
        return AlertPriority.HIGH
    if l in _MEDIUM_PRIORITY_LABELS:
        return AlertPriority.MEDIUM
    return AlertPriority.LOW


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------

@dataclass
class TrackAlertState:
    """
    Remembers what has already been announced for one tracked object.
    Reset when a track disappears and is later recreated.
    """

    track_id: int
    label: str

    last_alert_time: float = 0.0
    last_alert_type: Optional[AlertType] = None
    last_motion_state: Optional[str] = None
    last_distance_m: Optional[float] = None
    last_zone: DistanceZone = DistanceZone.UNKNOWN
    last_velocity_ms: Optional[float] = None

    frames_since_seen: int = 0      # incremented each frame the track is absent
    first_seen: bool = True         # True until the very first alert fires


# ---------------------------------------------------------------------------
# Alert event (output)
# ---------------------------------------------------------------------------

@dataclass
class AlertEvent:
    """
    A single alert produced for one tracked object.

    The frontend should speak `message` verbatim when `should_speak` is True.
    """

    track_id: int
    label: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    should_speak: bool
    distance_m: Optional[float] = None
    zone: DistanceZone = DistanceZone.UNKNOWN
    motion_state: Optional[str] = None
    velocity_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Stateful per-session alert decision engine.

    Call `process(tracked_objects, motion_results)` each frame.
    Returns a list of AlertEvents — only fire speech for those with
    `should_speak=True`.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = 3.0,
        zone_far_m: float = 5.0,
        zone_medium_m: float = 3.0,
        zone_near_m: float = 1.5,
        distance_change_threshold_m: float = 0.4,
        rapid_approach_threshold_ms: float = 1.5,
        track_disappear_frames: int = 10,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.zone_far_m = zone_far_m
        self.zone_medium_m = zone_medium_m
        self.zone_near_m = zone_near_m
        self.distance_change_threshold_m = distance_change_threshold_m
        self.rapid_approach_threshold_ms = rapid_approach_threshold_ms
        self.track_disappear_frames = track_disappear_frames

        # track_id → TrackAlertState
        self._states: Dict[int, TrackAlertState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        tracked_objects,      # List[TrackedDetection]
        motion_results: dict,  # track_id → MotionResult
        depths: dict,          # detection python-id → Optional[float]
        timestamp: float,
    ) -> List[AlertEvent]:
        """
        Evaluate all tracked objects and return alerts for this frame.

        Args:
            tracked_objects:  TrackedDetection list from the tracker.
            motion_results:   track_id → MotionResult from MotionAnalyzer.
            depths:           detection python-id → depth metres (or None).
            timestamp:        Unix timestamp of this frame.

        Returns:
            List of AlertEvent, sorted by priority (HIGH first).
        """
        now = timestamp
        active_ids = {td.track_id for td in tracked_objects}

        # Mark absent tracks
        for tid, state in list(self._states.items()):
            if tid not in active_ids:
                state.frames_since_seen += 1
                if state.frames_since_seen >= self.track_disappear_frames:
                    del self._states[tid]
                    logger.debug(f"AlertManager: dropped state for expired track {tid}")

        events: List[AlertEvent] = []

        for td in tracked_objects:
            tid  = td.track_id
            det  = td.detection
            mr   = motion_results.get(tid)
            depth_m = depths.get(id(det))

            # Get or create per-track state
            if tid not in self._states:
                self._states[tid] = TrackAlertState(
                    track_id=tid,
                    label=det.label,
                )

            state = self._states[tid]
            state.frames_since_seen = 0  # still alive

            event = self._evaluate(state, det, mr, depth_m, now)
            if event is not None:
                events.append(event)
                # Update state after firing
                state.last_alert_time  = now
                state.last_alert_type  = event.alert_type
                state.first_seen       = False
                if mr is not None:
                    state.last_motion_state = (
                        mr.state.value if hasattr(mr.state, "value") else str(mr.state)
                    )
                    if mr.velocity.value is not None:
                        state.last_velocity_ms = mr.velocity.value
                if depth_m is not None:
                    state.last_distance_m = depth_m
                    state.last_zone = self._zone(depth_m)

        # Sort: HIGH → MEDIUM → LOW
        _rank = {AlertPriority.HIGH: 0, AlertPriority.MEDIUM: 1, AlertPriority.LOW: 2}
        events.sort(key=lambda e: _rank.get(e.priority, 99))
        return events

    def reset(self) -> None:
        """Clear all per-track state (e.g. on session reset)."""
        self._states.clear()

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        state: TrackAlertState,
        det,
        mr,
        depth_m: Optional[float],
        now: float,
    ) -> Optional[AlertEvent]:
        """
        Decides whether to generate an alert for this object this frame.
        Returns an AlertEvent or None.
        """
        priority    = _object_priority(det.label)
        current_zone = self._zone(depth_m)
        motion_state = None
        velocity_ms  = None

        if mr is not None:
            motion_state = mr.state.value if hasattr(mr.state, "value") else str(mr.state)
            velocity_ms  = mr.velocity.value  # m/s, or None

        # ── Rule 1: new object (first time seen) ─────────────────────────
        if state.first_seen:
            return self._make_event(
                state, det, AlertType.NEW_OBJECT, priority,
                depth_m, current_zone, motion_state, velocity_ms,
                should_speak=(priority in (AlertPriority.HIGH, AlertPriority.MEDIUM)),
            )

        # ── Rule 2: rapid approach — always high priority ────────────────
        if (
            velocity_ms is not None
            and velocity_ms >= self.rapid_approach_threshold_ms
            and motion_state == "APPROACHING"
            and self._cooldown_elapsed(state, now, factor=0.5)  # shorter cooldown
        ):
            return self._make_event(
                state, det, AlertType.RAPID_APPROACH, AlertPriority.HIGH,
                depth_m, current_zone, motion_state, velocity_ms,
                should_speak=True,
            )

        # ── Rule 3: zone crossing ─────────────────────────────────────────
        if (
            current_zone != DistanceZone.UNKNOWN
            and current_zone != state.last_zone
            and state.last_zone != DistanceZone.UNKNOWN
            and self._cooldown_elapsed(state, now)
        ):
            return self._make_event(
                state, det, AlertType.ZONE_CROSSING, priority,
                depth_m, current_zone, motion_state, velocity_ms,
                should_speak=(priority in (AlertPriority.HIGH, AlertPriority.MEDIUM)),
            )

        # ── Rule 4: motion state change ──────────────────────────────────
        if (
            motion_state is not None
            and motion_state != "UNKNOWN"
            and motion_state != state.last_motion_state
            and state.last_motion_state is not None
            and self._cooldown_elapsed(state, now)
        ):
            return self._make_event(
                state, det, AlertType.MOTION_STATE_CHANGE, priority,
                depth_m, current_zone, motion_state, velocity_ms,
                should_speak=(priority in (AlertPriority.HIGH, AlertPriority.MEDIUM)),
            )

        # ── Rule 5: significant distance change within same zone ──────────
        if (
            depth_m is not None
            and state.last_distance_m is not None
            and abs(depth_m - state.last_distance_m) >= self.distance_change_threshold_m
            and self._cooldown_elapsed(state, now)
            and motion_state in ("APPROACHING", "MOVING_AWAY")
        ):
            return self._make_event(
                state, det, AlertType.DISTANCE_UPDATE, priority,
                depth_m, current_zone, motion_state, velocity_ms,
                should_speak=(
                    priority == AlertPriority.HIGH
                    or (priority == AlertPriority.MEDIUM and motion_state == "APPROACHING")
                ),
            )

        return None

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _make_event(
        self,
        state: TrackAlertState,
        det,
        alert_type: AlertType,
        priority: AlertPriority,
        depth_m: Optional[float],
        zone: DistanceZone,
        motion_state: Optional[str],
        velocity_ms: Optional[float],
        should_speak: bool,
    ) -> AlertEvent:
        message = self._build_message(
            det.label, alert_type, depth_m, zone, motion_state, velocity_ms
        )
        return AlertEvent(
            track_id=state.track_id,
            label=det.label,
            alert_type=alert_type,
            priority=priority,
            message=message,
            should_speak=should_speak,
            distance_m=depth_m,
            zone=zone,
            motion_state=motion_state,
            velocity_ms=velocity_ms,
        )

    def _build_message(
        self,
        label: str,
        alert_type: AlertType,
        depth_m: Optional[float],
        zone: DistanceZone,
        motion_state: Optional[str],
        velocity_ms: Optional[float],
    ) -> str:
        """
        Builds a natural-language alert message.
        The goal is a single compact sentence that is safe to speak aloud.
        """
        dist_str   = self._distance_str(depth_m, zone)
        motion_str = self._motion_str(motion_state, velocity_ms)

        if alert_type == AlertType.NEW_OBJECT:
            return f"{label.capitalize()} detected{dist_str}."

        if alert_type == AlertType.RAPID_APPROACH:
            speed = f", {velocity_ms:.1f} meters per second" if velocity_ms else ""
            return f"Warning! {label.capitalize()} coming rapidly{dist_str}{speed}."

        if alert_type == AlertType.ZONE_CROSSING:
            zone_label = self._zone_label(zone)
            return f"{label.capitalize()} now {zone_label}{dist_str}{motion_str}."

        if alert_type == AlertType.MOTION_STATE_CHANGE:
            return f"{label.capitalize()}{dist_str} — {self._motion_verb(motion_state)}."

        if alert_type == AlertType.DISTANCE_UPDATE:
            return f"{label.capitalize()}{dist_str}{motion_str}."

        if alert_type == AlertType.PERIODIC_REMINDER:
            return f"{label.capitalize()} still nearby{dist_str}{motion_str}."

        return f"{label.capitalize()} detected{dist_str}."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zone(self, depth_m: Optional[float]) -> DistanceZone:
        if depth_m is None:
            return DistanceZone.UNKNOWN
        if depth_m <= self.zone_near_m:
            return DistanceZone.VERY_NEAR
        if depth_m <= self.zone_medium_m:
            return DistanceZone.NEAR
        if depth_m <= self.zone_far_m:
            return DistanceZone.MEDIUM
        return DistanceZone.FAR

    def _zone_label(self, zone: DistanceZone) -> str:
        return {
            DistanceZone.VERY_NEAR: "very close",
            DistanceZone.NEAR:      "nearby",
            DistanceZone.MEDIUM:    "at medium distance",
            DistanceZone.FAR:       "far away",
            DistanceZone.UNKNOWN:   "detected",
        }.get(zone, "detected")

    def _distance_str(self, depth_m: Optional[float], zone: DistanceZone) -> str:
        if depth_m is not None:
            return f", approximately {depth_m:.1f} meters away"
        # Qualitative fallback when no metric depth
        if zone != DistanceZone.UNKNOWN:
            return f", {self._zone_label(zone)}"
        return ""

    def _motion_str(self, motion_state: Optional[str], velocity_ms: Optional[float]) -> str:
        if motion_state == "APPROACHING":
            if velocity_ms is not None:
                return f", coming at {velocity_ms:.1f} m/s"
            return ", coming"
        if motion_state == "MOVING_AWAY":
            return ", going away"
        if motion_state == "STATIONARY":
            return ", stationary"
        return ""

    def _motion_verb(self, motion_state: Optional[str]) -> str:
        return {
            "APPROACHING":  "now coming",
            "MOVING_AWAY":  "going away",
            "STATIONARY":   "now stationary",
        }.get(motion_state or "", "status changed")

    def _cooldown_elapsed(
        self, state: TrackAlertState, now: float, factor: float = 1.0
    ) -> bool:
        """Returns True if enough time has passed since the last alert."""
        elapsed = now - state.last_alert_time
        return elapsed >= self.min_interval_s * factor
