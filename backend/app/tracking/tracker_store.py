"""
Walking Eye - AI Perception Engine
Tracker Store.

A process-level singleton that maps session_id → SessionState.
Each session holds its own independent tracker, temporal buffers,
motion analyser, and alert manager — so multiple concurrent Flutter
clients do not share state.

Design decisions
----------------
* Created once at startup, stored in app.state.
* MAX_TRACKER_SESSIONS caps memory: oldest (LRU) session evicted at limit.
* Sessions idle > SESSION_IDLE_TIMEOUT_S are pruned automatically.
* threading.Lock protects all mutations.

process_frame() now returns a 3-tuple:
    (tracked_detections, motion_results, alert_events)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.alerting.alert_manager import AlertEvent, AlertManager
from app.tracking.motion_analyzer import MotionAnalyzer, MotionResult
from app.tracking.temporal_buffer import Observation, TemporalBuffer
from app.tracking.tracker import IoUTracker
from app.utilities.logger import get_logger

logger = get_logger(__name__)

SESSION_IDLE_TIMEOUT_S: float = 300.0   # 5 minutes


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """
    All per-session tracking state:
      tracker         — IoUTracker (persistent track IDs)
      buffers         — track_id → TemporalBuffer
      motion_analyzer — stateless motion classifier
      alert_manager   — stateful alert deduplication engine
      last_seen       — Unix timestamp for LRU eviction
      frame_count     — monotonically increasing frame counter
    """

    tracker: IoUTracker
    buffers: Dict[int, TemporalBuffer] = field(default_factory=dict)
    motion_analyzer: MotionAnalyzer = field(default_factory=MotionAnalyzer)
    alert_manager: AlertManager = field(default_factory=AlertManager)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0

    def get_or_create_buffer(self, track_id: int, history_size: int) -> TemporalBuffer:
        if track_id not in self.buffers:
            self.buffers[track_id] = TemporalBuffer(history_size=history_size)
            logger.debug(f"Created TemporalBuffer for track_id={track_id}")
        return self.buffers[track_id]

    def purge_dead_tracks(self, active_track_ids: list) -> None:
        active_set = set(active_track_ids)
        dead = [tid for tid in list(self.buffers.keys()) if tid not in active_set]
        for tid in dead:
            del self.buffers[tid]
            logger.debug(f"Purged TemporalBuffer for expired track_id={tid}")


# ---------------------------------------------------------------------------
# Tracker store
# ---------------------------------------------------------------------------

class TrackerStore:
    """
    Process-level singleton store for all tracking sessions.

    Constructor kwargs cover both the motion analyser and the alert manager
    so the caller (main.py lifespan) can drive everything from Settings.
    """

    def __init__(
        self,
        *,
        # --- Session limits ---
        max_sessions: int = 50,
        # --- Tracker ---
        history_size: int = 5,
        iou_threshold: float = 0.30,
        max_age: int = 5,
        # --- Motion analyser ---
        min_track_history: int = 2,
        stationary_depth_threshold: float = 0.08,
        approaching_depth_threshold: float = 0.08,
        stationary_scale_threshold: float = 0.05,
        approaching_scale_threshold: float = 0.05,
        direction_noise_threshold_px: int = 5,
        # --- Alert manager ---
        alert_min_interval_s: float = 3.0,
        alert_zone_far_m: float = 5.0,
        alert_zone_medium_m: float = 3.0,
        alert_zone_near_m: float = 1.5,
        alert_distance_change_threshold_m: float = 0.4,
        alert_rapid_approach_threshold_ms: float = 1.5,
        alert_track_disappear_frames: int = 10,
    ) -> None:
        self._max_sessions = max_sessions
        self._history_size = history_size
        self._iou_threshold = iou_threshold
        self._max_age = max_age

        self._analyzer_kwargs = dict(
            min_track_history=min_track_history,
            stationary_depth_threshold=stationary_depth_threshold,
            approaching_depth_threshold=approaching_depth_threshold,
            stationary_scale_threshold=stationary_scale_threshold,
            approaching_scale_threshold=approaching_scale_threshold,
            direction_noise_threshold_px=direction_noise_threshold_px,
        )

        self._alert_kwargs = dict(
            min_interval_s=alert_min_interval_s,
            zone_far_m=alert_zone_far_m,
            zone_medium_m=alert_zone_medium_m,
            zone_near_m=alert_zone_near_m,
            distance_change_threshold_m=alert_distance_change_threshold_m,
            rapid_approach_threshold_ms=alert_rapid_approach_threshold_ms,
            track_disappear_frames=alert_track_disappear_frames,
        )

        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

        logger.info(
            f"TrackerStore initialised: max_sessions={max_sessions}, "
            f"history_size={history_size}, iou_threshold={iou_threshold}, "
            f"max_age={max_age}, alert_min_interval_s={alert_min_interval_s}"
        )

    # ------------------------------------------------------------------
    # Session access
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> SessionState:
        with self._lock:
            self._evict_idle_sessions()

            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                session = self._sessions[session_id]
                session.last_seen = time.time()
                return session

            if len(self._sessions) >= self._max_sessions:
                oldest_id, _ = next(iter(self._sessions.items()))
                del self._sessions[oldest_id]
                logger.info(f"TrackerStore: evicted oldest session '{oldest_id}'")

            session = SessionState(
                tracker=IoUTracker(
                    iou_threshold=self._iou_threshold,
                    max_age=self._max_age,
                ),
                motion_analyzer=MotionAnalyzer(**self._analyzer_kwargs),
                alert_manager=AlertManager(**self._alert_kwargs),
            )
            self._sessions[session_id] = session
            logger.info(
                f"TrackerStore: created session '{session_id}' "
                f"(total: {len(self._sessions)})"
            )
            return session

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"TrackerStore: removed session '{session_id}'")

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def process_frame(
        self,
        session_id: str,
        detections,
        timestamp: float,
        estimated_depths=None,
    ):
        """
        Runs the full pipeline for one frame:
            YOLO detections → tracker → temporal buffers → motion analysis → alert manager

        Returns:
            (tracked_detections, motion_results, alert_events)

            tracked_detections  — List[TrackedDetection]
            motion_results      — Dict[track_id, MotionResult]
            alert_events        — List[AlertEvent] (sorted HIGH → MEDIUM → LOW)
        """
        session = self.get_or_create_session(session_id)
        session.frame_count += 1
        frame_index = session.frame_count

        # --- Tracker ---
        tracked = session.tracker.update(detections, frame_index, timestamp)

        # Map detection python-id → depth for fast lookup
        depth_by_det_id: dict = {}
        if estimated_depths and len(estimated_depths) == len(detections):
            for i, det in enumerate(detections):
                depth_by_det_id[id(det)] = estimated_depths[i]

        # --- Push observations into temporal buffers ---
        for td in tracked:
            tid = td.track_id
            det = td.detection
            buf = session.get_or_create_buffer(tid, self._history_size)
            depth = depth_by_det_id.get(id(det))

            obs = Observation.from_values(
                track_id=tid,
                timestamp=timestamp,
                frame_index=frame_index,
                bbox_x=det.x,
                bbox_y=det.y,
                bbox_width=det.width,
                bbox_height=det.height,
                confidence=det.confidence,
                estimated_depth=depth,
            )
            buf.push(obs)

        # --- Prune expired track buffers ---
        session.purge_dead_tracks(session.tracker.active_track_ids)

        # --- Motion analysis ---
        motion_results: dict = {}
        for td in tracked:
            tid = td.track_id
            buf = session.buffers.get(tid)
            if buf:
                motion_results[tid] = session.motion_analyzer.analyze(buf)

        # --- Alert manager ---
        alert_events: List[AlertEvent] = session.alert_manager.process(
            tracked_objects=tracked,
            motion_results=motion_results,
            depths=depth_by_det_id,
            timestamp=timestamp,
        )

        return tracked, motion_results, alert_events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_idle_sessions(self) -> None:
        now = time.time()
        idle_ids = [
            sid for sid, sess in self._sessions.items()
            if (now - sess.last_seen) > SESSION_IDLE_TIMEOUT_S
        ]
        for sid in idle_ids:
            del self._sessions[sid]
            logger.info(f"TrackerStore: evicted idle session '{sid}'")
