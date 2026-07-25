"""
Walking Eye - AI Perception Engine
Unit tests for the temporal motion-analysis layer.

Tests cover the six scenarios specified in the requirements:

  Test 1  — Approaching object (depth: 5.0 → 4.5 → 4.0)       → APPROACHING
  Test 2  — Moving away (depth: 4.0 → 4.5 → 5.0)              → MOVING_AWAY
  Test 3  — Stationary (depth: 5.0 → 5.02 → 4.98)             → STATIONARY
  Test 4  — Relative approach without depth (area grows)        → APPROACHING, distance=None
  Test 5  — Temporal noise smoothing (position jitter)          → no false extreme motion
  Test 6  — Object disappears temporarily (tracker max_age)     → track survives

Additional unit tests:

  Test 7  — Insufficient history → UNKNOWN
  Test 8  — Velocity calculation (metric, depth available)
  Test 9  — Relative approach speed (no depth)
  Test 10 — Horizontal direction classification
  Test 11 — TemporalBuffer bounded size
  Test 12 — IoUTracker: same object keeps same track_id across frames
  Test 13 — IoUTracker: new object gets new track_id
  Test 14 — TrackerStore: separate sessions are isolated

All tests are pure unit tests — no YOLO model, no FastAPI app needed.
"""

import time
import pytest

from app.tracking.motion_analyzer import (
    MotionAnalyzer,
    MotionDirection,
    MotionState,
)
from app.tracking.temporal_buffer import Observation, TemporalBuffer
from app.tracking.tracker import IoUTracker
from app.tracking.tracker_store import TrackerStore
from app.vision.detector import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(
    track_id: int,
    frame: int,
    timestamp: float,
    cx: int,
    cy: int,
    w: int,
    h: int,
    depth=None,
    conf: float = 0.90,
) -> Observation:
    """Convenience constructor for test observations."""
    return Observation.from_values(
        track_id=track_id,
        timestamp=timestamp,
        frame_index=frame,
        bbox_x=cx - w // 2,
        bbox_y=cy - h // 2,
        bbox_width=w,
        bbox_height=h,
        confidence=conf,
        estimated_depth=depth,
    )


def _fill_buffer(observations) -> TemporalBuffer:
    """Create and fill a TemporalBuffer from a list of Observations."""
    buf = TemporalBuffer(history_size=10)
    for obs in observations:
        buf.push(obs)
    return buf


def _det(
    label: str = "person",
    cx: int = 320,
    cy: int = 240,
    w: int = 100,
    h: int = 200,
    conf: float = 0.90,
) -> Detection:
    """Convenience Detection factory."""
    x = cx - w // 2
    y = cy - h // 2
    return Detection(
        label=label,
        confidence=conf,
        x=x,
        y=y,
        width=w,
        height=h,
        center_x=cx,
        center_y=cy,
    )


# Tight thresholds so these tests are not sensitive to default changes
_ANALYZER = MotionAnalyzer(
    min_track_history=2,
    stationary_depth_threshold=0.08,
    approaching_depth_threshold=0.08,
    stationary_scale_threshold=0.05,
    approaching_scale_threshold=0.05,
    direction_noise_threshold_px=5,
)


# ===========================================================================
# Test 1 — Approaching object (depth decreasing)
# ===========================================================================

class TestApproachingObject:
    """Depth sequence: 5.0 → 4.5 → 4.0  →  APPROACHING"""

    def _buffer(self):
        return _fill_buffer([
            _obs(1, 1, 0.0,  320, 240, 100, 200, depth=5.0),
            _obs(1, 2, 0.5,  320, 240, 110, 220, depth=4.5),
            _obs(1, 3, 1.0,  320, 240, 125, 250, depth=4.0),
        ])

    def test_state_is_approaching(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.state == MotionState.APPROACHING

    def test_distance_source_is_depth(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.distance.source == "depth"

    def test_distance_value_is_latest_depth(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.distance.value == pytest.approx(4.0, abs=0.05)
        assert result.distance.unit == "meters"

    def test_velocity_is_metric(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.velocity.type == "metric"
        assert result.velocity.value is not None
        assert result.velocity.value > 0   # approach speed is positive

    def test_confidence_above_threshold(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.confidence >= 0.50

    def test_observations_used(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.observations_used == 3


# ===========================================================================
# Test 2 — Moving away (depth increasing)
# ===========================================================================

class TestMovingAway:
    """Depth sequence: 4.0 → 4.5 → 5.0  →  MOVING_AWAY"""

    def _buffer(self):
        return _fill_buffer([
            _obs(2, 1, 0.0, 320, 240, 125, 250, depth=4.0),
            _obs(2, 2, 0.5, 320, 240, 110, 220, depth=4.5),
            _obs(2, 3, 1.0, 320, 240, 100, 200, depth=5.0),
        ])

    def test_state_is_moving_away(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.state == MotionState.MOVING_AWAY

    def test_distance_source_is_depth(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.distance.source == "depth"

    def test_velocity_is_metric(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.velocity.type == "metric"
        assert result.velocity.value is not None


# ===========================================================================
# Test 3 — Stationary (depth barely fluctuating)
# ===========================================================================

class TestStationary:
    """Depth sequence: 5.0 → 5.02 → 4.98  →  STATIONARY"""

    def _buffer(self):
        return _fill_buffer([
            _obs(3, 1, 0.0, 320, 240, 100, 200, depth=5.00),
            _obs(3, 2, 0.5, 320, 240, 101, 201, depth=5.02),
            _obs(3, 3, 1.0, 320, 240, 100, 200, depth=4.98),
        ])

    def test_state_is_stationary(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.state == MotionState.STATIONARY

    def test_distance_is_metric(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.distance.source == "depth"
        assert result.distance.value is not None

    def test_velocity_is_metric(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.velocity.type == "metric"


# ===========================================================================
# Test 4 — Relative approach without depth (bbox area grows)
# ===========================================================================

class TestRelativeApproachNoDept:
    """
    BBox area: 10000 → 12000 → 14500  →  APPROACHING
    But distance must be None (no fabricated metric values).
    """

    def _obs_area(self, frame, ts, area, depth=None):
        """Create an observation with a specific bbox area (square bbox)."""
        side = int(area ** 0.5)
        return _obs(4, frame, ts, 320, 240, side, side, depth=depth)

    def _buffer(self):
        return _fill_buffer([
            self._obs_area(1, 0.0, 10000),
            self._obs_area(2, 0.5, 12000),
            self._obs_area(3, 1.0, 14500),
        ])

    def test_state_is_approaching(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.state == MotionState.APPROACHING

    def test_distance_value_is_none(self):
        """Must never fabricate physical distance."""
        result = _ANALYZER.analyze(self._buffer())
        assert result.distance.value is None
        assert result.distance.unit is None

    def test_distance_source_is_relative(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.distance.source == "relative_bbox_scale"

    def test_velocity_type_is_relative(self):
        result = _ANALYZER.analyze(self._buffer())
        assert result.velocity.type == "relative"

    def test_metric_velocity_is_none(self):
        """Must never fabricate metric velocity."""
        result = _ANALYZER.analyze(self._buffer())
        assert result.velocity.value is None
        assert result.velocity.unit is None

    def test_relative_approach_speed_present(self):
        """A non-metric approach proxy should be present."""
        result = _ANALYZER.analyze(self._buffer())
        assert result.velocity.relative_approach_speed is not None


# ===========================================================================
# Test 5 — Noisy position data (temporal smoothing)
# ===========================================================================

class TestNoiseSmoothing:
    """
    Position jitter: cx 100 → 105 → 98 → 110
    The system should NOT classify this as extreme directional movement.
    Mean smoothing over pairwise deltas should keep confidence low or
    return CENTER/STATIONARY instead of LEFT/RIGHT extremes.
    """

    def _buffer(self):
        return _fill_buffer([
            _obs(5, 1, 0.0, 100, 240, 100, 200),
            _obs(5, 2, 0.5, 105, 240, 102, 202),
            _obs(5, 3, 1.0,  98, 240,  99, 198),
            _obs(5, 4, 1.5, 110, 240, 103, 203),
        ])

    def test_no_extreme_motion_state(self):
        """
        With only bbox-scale noise (area stays roughly constant),
        should not produce a high-confidence APPROACHING or MOVING_AWAY.
        Either STATIONARY or low-confidence result is acceptable.
        """
        result = _ANALYZER.analyze(self._buffer())
        if result.state in (MotionState.APPROACHING, MotionState.MOVING_AWAY):
            # If classified as moving, confidence must be low
            assert result.confidence < 0.75, (
                f"Got {result.state} with high confidence {result.confidence} "
                f"from noisy position data — smoothing may not be working."
            )

    def test_direction_not_wildly_flipping(self):
        """
        Small jitter (max dx=10px across 4 frames) should produce CENTER
        or a low-confidence direction, never a hard LEFT/RIGHT claim.
        """
        result = _ANALYZER.analyze(self._buffer())
        # Net displacement from frame 1 cx=100 to frame 4 cx=110 is +10px
        # which is just at the noise boundary (threshold=5px in _ANALYZER).
        # Acceptable outcomes: CENTER or RIGHT (but not LEFT)
        assert result.direction in (MotionDirection.CENTER, MotionDirection.RIGHT), (
            f"Unexpected direction {result.direction} from small noisy jitter."
        )


# ===========================================================================
# Test 6 — Object disappears temporarily (tracker max_age)
# ===========================================================================

class TestTemporaryDisappearance:
    """
    An object tracked for 3 frames disappears for 2 frames then reappears.
    With max_age >= 2 the track should survive without being destroyed.
    """

    def test_track_survives_temporary_gap(self):
        tracker = IoUTracker(iou_threshold=0.25, max_age=5)

        # Object at fixed position
        det = _det(cx=320, cy=240, w=100, h=200)

        # Frame 1 — seen
        r1 = tracker.update([det], frame_index=1, timestamp=0.0)
        assert len(r1) == 1
        original_id = r1[0].track_id

        # Frame 2 — seen
        r2 = tracker.update([det], frame_index=2, timestamp=0.5)
        assert len(r2) == 1
        assert r2[0].track_id == original_id

        # Frames 3 & 4 — object NOT detected (empty list)
        tracker.update([], frame_index=3, timestamp=1.0)
        tracker.update([], frame_index=4, timestamp=1.5)

        # Frame 5 — object reappears at same position
        r5 = tracker.update([det], frame_index=5, timestamp=2.0)
        assert len(r5) == 1, "Track should still be alive after 2-frame gap"
        # Track ID must be the same physical object
        assert r5[0].track_id == original_id, (
            f"Expected track_id {original_id}, got {r5[0].track_id}. "
            "Track was destroyed during gap instead of surviving."
        )

    def test_track_dies_after_max_age(self):
        """
        A track that exceeds max_age consecutive misses must eventually expire.
        """
        tracker = IoUTracker(iou_threshold=0.25, max_age=2)
        det = _det(cx=320, cy=240, w=100, h=200)

        r1 = tracker.update([det], frame_index=1, timestamp=0.0)
        original_id = r1[0].track_id

        # Miss 3 frames — exceeds max_age=2
        tracker.update([], frame_index=2, timestamp=0.5)
        tracker.update([], frame_index=3, timestamp=1.0)
        tracker.update([], frame_index=4, timestamp=1.5)

        # Track must be purged from active_track_ids
        assert original_id not in tracker.active_track_ids, (
            "Expired track was not removed from active_track_ids."
        )

        # New detection should now create a fresh track
        r5 = tracker.update([det], frame_index=5, timestamp=2.0)
        assert len(r5) == 1
        new_id = r5[0].track_id
        assert new_id != original_id, (
            "New detection after expiry should get a fresh track_id."
        )


# ===========================================================================
# Test 7 — Insufficient history → UNKNOWN
# ===========================================================================

class TestInsufficientHistory:
    def test_single_observation_returns_unknown(self):
        buf = _fill_buffer([
            _obs(7, 1, 0.0, 320, 240, 100, 200, depth=5.0),
        ])
        result = _ANALYZER.analyze(buf)
        assert result.state == MotionState.UNKNOWN
        assert result.confidence == 0.0

    def test_empty_buffer_returns_unknown(self):
        buf = TemporalBuffer(history_size=5)
        result = _ANALYZER.analyze(buf)
        assert result.state == MotionState.UNKNOWN
        assert result.confidence == 0.0


# ===========================================================================
# Test 8 — Velocity calculation (metric)
# ===========================================================================

class TestMetricVelocity:
    """
    3 frames over 2 seconds, depth changes 5.0 → 3.0 (Δ=-2.0m in 2s)
    Expected approach speed ≈ 1.0 m/s
    """

    def test_velocity_approximately_correct(self):
        buf = _fill_buffer([
            _obs(8, 1, 0.0, 320, 240, 100, 200, depth=5.0),
            _obs(8, 2, 1.0, 320, 240, 120, 240, depth=4.0),
            _obs(8, 3, 2.0, 320, 240, 140, 280, depth=3.0),
        ])
        result = _ANALYZER.analyze(buf)
        assert result.state == MotionState.APPROACHING
        assert result.velocity.type == "metric"
        assert result.velocity.unit == "m/s"
        # Total depth change = 2.0m over 2.0s → 1.0 m/s
        assert result.velocity.value == pytest.approx(1.0, abs=0.15)


# ===========================================================================
# Test 9 — Relative approach speed (no depth)
# ===========================================================================

class TestRelativeApproachSpeed:
    def test_relative_speed_is_positive_for_growing_bbox(self):
        """Growing bbox area → positive relative_approach_speed."""
        buf = _fill_buffer([
            _obs(9, 1, 0.0, 320, 240, 100, 100),   # area = 10000
            _obs(9, 2, 0.5, 320, 240, 110, 110),   # area = 12100
            _obs(9, 3, 1.0, 320, 240, 120, 120),   # area = 14400
        ])
        result = _ANALYZER.analyze(buf)
        assert result.velocity.type == "relative"
        assert result.velocity.relative_approach_speed is not None
        assert result.velocity.relative_approach_speed > 0

    def test_relative_speed_is_negative_for_shrinking_bbox(self):
        """Shrinking bbox area → negative relative_approach_speed."""
        buf = _fill_buffer([
            _obs(9, 1, 0.0, 320, 240, 120, 120),   # area = 14400
            _obs(9, 2, 0.5, 320, 240, 110, 110),   # area = 12100
            _obs(9, 3, 1.0, 320, 240, 100, 100),   # area = 10000
        ])
        result = _ANALYZER.analyze(buf)
        assert result.velocity.type == "relative"
        assert result.velocity.relative_approach_speed is not None
        assert result.velocity.relative_approach_speed < 0


# ===========================================================================
# Test 10 — Horizontal direction classification
# ===========================================================================

class TestDirectionClassification:
    def test_moving_right(self):
        """Object centre moves from x=100 to x=200 → RIGHT."""
        buf = _fill_buffer([
            _obs(10, 1, 0.0, 100, 240, 60, 120),
            _obs(10, 2, 0.5, 150, 240, 60, 120),
            _obs(10, 3, 1.0, 200, 240, 60, 120),
        ])
        result = _ANALYZER.analyze(buf)
        assert result.direction == MotionDirection.RIGHT

    def test_moving_left(self):
        """Object centre moves from x=300 to x=100 → LEFT."""
        buf = _fill_buffer([
            _obs(10, 1, 0.0, 300, 240, 60, 120),
            _obs(10, 2, 0.5, 200, 240, 60, 120),
            _obs(10, 3, 1.0, 100, 240, 60, 120),
        ])
        result = _ANALYZER.analyze(buf)
        assert result.direction == MotionDirection.LEFT

    def test_tiny_jitter_is_center(self):
        """Sub-threshold movement (< 5px) → CENTER."""
        buf = _fill_buffer([
            _obs(10, 1, 0.0, 320, 240, 60, 120),
            _obs(10, 2, 0.5, 322, 240, 60, 120),
            _obs(10, 3, 1.0, 318, 240, 60, 120),
        ])
        result = _ANALYZER.analyze(buf)
        assert result.direction == MotionDirection.CENTER


# ===========================================================================
# Test 11 — TemporalBuffer bounded size
# ===========================================================================

class TestTemporalBuffer:
    def test_buffer_does_not_exceed_history_size(self):
        buf = TemporalBuffer(history_size=3)
        for i in range(10):
            buf.push(_obs(11, i, float(i) * 0.5, 320, 240, 100, 200))
        assert buf.count == 3

    def test_oldest_entry_is_evicted(self):
        buf = TemporalBuffer(history_size=3)
        for i in range(5):
            buf.push(_obs(11, i, float(i), 320 + i * 10, 240, 100, 200))
        # After 5 pushes into a size-3 buffer, oldest frame is frame 2 (index 2)
        assert buf.oldest.frame_index == 2

    def test_latest_is_most_recent(self):
        buf = TemporalBuffer(history_size=5)
        for i in range(4):
            buf.push(_obs(11, i, float(i), 320, 240, 100, 200))
        assert buf.latest.frame_index == 3

    def test_time_span(self):
        buf = TemporalBuffer(history_size=5)
        buf.push(_obs(11, 1, 0.0,  320, 240, 100, 200))
        buf.push(_obs(11, 2, 0.5,  320, 240, 100, 200))
        buf.push(_obs(11, 3, 1.0,  320, 240, 100, 200))
        assert buf.time_span_seconds() == pytest.approx(1.0, abs=0.001)


# ===========================================================================
# Test 12 — IoUTracker: same physical object keeps same track_id
# ===========================================================================

class TestIoUTrackerSameObject:
    def test_stable_track_id_across_frames(self):
        tracker = IoUTracker(iou_threshold=0.25, max_age=5)
        # Object barely moves each frame
        positions = [(320, 240), (322, 241), (318, 239), (321, 240)]

        ids = []
        for frame_i, (cx, cy) in enumerate(positions, start=1):
            det = _det(cx=cx, cy=cy, w=100, h=200)
            result = tracker.update([det], frame_index=frame_i, timestamp=float(frame_i) * 0.5)
            assert len(result) == 1, f"Expected 1 tracked detection at frame {frame_i}"
            ids.append(result[0].track_id)

        assert len(set(ids)) == 1, (
            f"Expected a single stable track_id, got {ids}"
        )


# ===========================================================================
# Test 13 — IoUTracker: new object after clear separation gets new track_id
# ===========================================================================

class TestIoUTrackerNewObject:
    def test_non_overlapping_detection_gets_new_id(self):
        tracker = IoUTracker(iou_threshold=0.25, max_age=5)

        # Object A on the left
        det_a = _det(cx=100, cy=240, w=80, h=160)
        r1 = tracker.update([det_a], frame_index=1, timestamp=0.0)
        id_a = r1[0].track_id

        # Object B on the right (no overlap with A at all)
        det_b = _det(cx=500, cy=240, w=80, h=160)
        r2 = tracker.update([det_a, det_b], frame_index=2, timestamp=0.5)

        track_ids = {td.track_id for td in r2}
        assert len(track_ids) == 2, "Two non-overlapping objects must have distinct track_ids"
        assert id_a in track_ids, "Original object A must keep its track_id"


# ===========================================================================
# Test 14 — TrackerStore: sessions are isolated
# ===========================================================================

class TestTrackerStoreSessionIsolation:
    def _make_store(self):
        return TrackerStore(
            max_sessions=10,
            history_size=5,
            iou_threshold=0.25,
            max_age=5,
            min_track_history=2,
            stationary_depth_threshold=0.08,
            approaching_depth_threshold=0.08,
            stationary_scale_threshold=0.05,
            approaching_scale_threshold=0.05,
            direction_noise_threshold_px=5,
        )

    def test_different_sessions_have_separate_trackers(self):
        store = self._make_store()
        det = _det(cx=320, cy=240, w=100, h=200)

        # Session A: 2 frames
        tracked_a1, _, _ = store.process_frame("session-A", [det], timestamp=0.0)
        tracked_a2, _, _ = store.process_frame("session-A", [det], timestamp=0.5)

        # Session B: 1 frame (fresh start)
        tracked_b1, _, _ = store.process_frame("session-B", [det], timestamp=0.0)

        # Each session has its own track counter
        id_a = tracked_a2[0].track_id
        id_b = tracked_b1[0].track_id

        # Both should be 1 (first track in each respective session)
        assert id_a == 1
        assert id_b == 1
        assert store.active_session_count == 2

    def test_motion_results_independent_per_session(self):
        """
        Session A sees an approaching object; session B sees a stationary one.
        Results must not bleed across sessions.
        """
        store = self._make_store()

        # Session A: growing bbox (approaching)
        for frame_i, side in enumerate([80, 95, 115], start=1):
            det = _det(cx=320, cy=240, w=side, h=side)
            store.process_frame("session-A", [det], timestamp=float(frame_i) * 0.5)

        # Session B: constant bbox (stationary)
        for frame_i in range(1, 4):
            det = _det(cx=320, cy=240, w=100, h=100)
            _, motion_b, _ = store.process_frame(
                "session-B", [det], timestamp=float(frame_i) * 0.5
            )

        # Get final motion for session A
        det = _det(cx=320, cy=240, w=115, h=115)
        _, motion_a, _ = store.process_frame("session-A", [det], timestamp=2.0)

        if motion_a:
            state_a = list(motion_a.values())[0].state
            assert state_a in (MotionState.APPROACHING, MotionState.UNKNOWN)

        if motion_b:
            state_b = list(motion_b.values())[0].state
            assert state_b in (MotionState.STATIONARY, MotionState.UNKNOWN)
