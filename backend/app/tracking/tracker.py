"""
Walking Eye - AI Perception Engine
IoU Tracker with Kalman Filtering.

A lightweight multi-object tracker inspired by ByteTrack / SORT principles,
implemented without any external tracking library so the package list stays
minimal.  The tracker runs entirely on numpy — no PyTorch required.

Design decisions
----------------
* Each Detection that arrives is matched to an existing Track via IoU.
  If no match exceeds the configured threshold a new Track is created.
* Each Track owns a tiny Kalman Filter (constant-velocity model) that
  predicts the next bounding-box position and smooths noisy observations.
* Unmatched tracks survive up to `max_age` consecutive frames (so a
  brief occlusion does not immediately kill a track).
* The public API is intentionally simple:
      tracks = tracker.update(detections, frame_index, timestamp)
  which returns TrackedDetection objects — Detection + track_id.

Kalman state vector  [cx, cy, w, h, vcx, vcy, vw, vh]
  cx, cy = bounding-box centre
  w, h   = bounding-box width and height
  v*     = velocities (first derivatives)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.utilities.logger import get_logger
from app.vision.detector import Detection

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Kalman filter matrices  (constant-velocity model, 8-dimensional state)
# ---------------------------------------------------------------------------

_DIM_X = 8   # state:   [cx, cy, w, h, vcx, vcy, vw, vh]
_DIM_Z = 4   # measure: [cx, cy, w, h]

# State-transition matrix F  (x_{k} = F * x_{k-1})
_F = np.eye(_DIM_X, dtype=np.float64)
for i in range(_DIM_Z):
    _F[i, i + _DIM_Z] = 1.0          # cx += vcx, etc.

# Measurement matrix H  (z = H * x)
_H = np.zeros((_DIM_Z, _DIM_X), dtype=np.float64)
for i in range(_DIM_Z):
    _H[i, i] = 1.0

# Process-noise covariance Q
_Q = np.eye(_DIM_X, dtype=np.float64)
_Q[4, 4] = _Q[5, 5] = 1e-2   # velocity noise
_Q[6, 6] = _Q[7, 7] = 1e-4

# Measurement-noise covariance R
_R = np.eye(_DIM_Z, dtype=np.float64) * 4.0

# Initial state-error covariance P
_P_INIT = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4, 1e4])


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackedDetection:
    """A Detection annotated with a persistent track ID."""

    detection: Detection
    track_id: int


@dataclass
class Track:
    """
    A single multi-frame track.

    Maintains Kalman-filter state and bookkeeping (age, hits, misses).
    """

    track_id: int
    hits: int = 1
    age: int = 1
    consecutive_misses: int = 0
    is_confirmed: bool = False  # promoted after MIN_HITS observations

    # Kalman state and covariance
    x: np.ndarray = field(default_factory=lambda: np.zeros((_DIM_X, 1)))
    P: np.ndarray = field(default_factory=lambda: _P_INIT.copy())

    # Last known label (for identity continuity across misses)
    label: str = ""

    # -----------------------------------------------------------------------
    # Kalman methods
    # -----------------------------------------------------------------------

    def predict(self) -> None:
        """Propagate state forward by one time step."""
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        self.age += 1
        self.consecutive_misses += 1

    def update(self, bbox_cx_cy_w_h: np.ndarray) -> None:
        """Correct state with a new measurement."""
        z = bbox_cx_cy_w_h.reshape((_DIM_Z, 1))
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(_DIM_X) - K @ _H) @ self.P
        self.hits += 1
        self.consecutive_misses = 0

    @property
    def predicted_bbox(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h) from the current Kalman state."""
        cx, cy, w, h = self.x[:4, 0]
        w = max(1.0, w)
        h = max(1.0, h)
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return x, y, int(w), int(h)

    @classmethod
    def from_detection(cls, detection: Detection, track_id: int) -> "Track":
        """Initialise a brand-new track from a detection."""
        cx = detection.center_x
        cy = detection.center_y
        w = detection.width
        h = detection.height
        x0 = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        t = cls(track_id=track_id, label=detection.label)
        t.x = x0
        t.P = _P_INIT.copy()
        return t


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _bbox_to_tlbr(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Convert (x, y, w, h) → [x1, y1, x2, y2]."""
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    Both arrays must be 1-D length-4 vectors.
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def _build_iou_matrix(
    tracks: List[Track],
    detections: List[Detection],
) -> np.ndarray:
    """
    Returns an (N_tracks × N_detections) IoU matrix.
    Tracks supply their Kalman-predicted bboxes; detections supply observed bboxes.
    """
    n_t = len(tracks)
    n_d = len(detections)
    mat = np.zeros((n_t, n_d), dtype=np.float64)

    for ti, track in enumerate(tracks):
        tx, ty, tw, th = track.predicted_bbox
        t_box = _bbox_to_tlbr(tx, ty, tw, th)
        for di, det in enumerate(detections):
            d_box = _bbox_to_tlbr(det.x, det.y, det.width, det.height)
            mat[ti, di] = _iou(t_box, d_box)

    return mat


# ---------------------------------------------------------------------------
# Hungarian / greedy matching
# ---------------------------------------------------------------------------

def _greedy_match(
    iou_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedily matches tracks to detections using the highest-IoU pair first.

    Returns:
        matched:          list of (track_idx, det_idx) pairs
        unmatched_tracks: list of track indices with no match
        unmatched_dets:   list of detection indices with no match
    """
    if iou_matrix.size == 0:
        n_t, n_d = iou_matrix.shape if iou_matrix.ndim == 2 else (0, 0)
        return [], list(range(n_t)), list(range(n_d))

    matched: List[Tuple[int, int]] = []
    assigned_tracks = set()
    assigned_dets = set()

    # Flatten and sort by IoU descending for greedy assignment
    flat = [(iou_matrix[ti, di], ti, di)
            for ti in range(iou_matrix.shape[0])
            for di in range(iou_matrix.shape[1])]
    flat.sort(reverse=True, key=lambda x: x[0])

    for score, ti, di in flat:
        if score < threshold:
            break
        if ti in assigned_tracks or di in assigned_dets:
            continue
        matched.append((ti, di))
        assigned_tracks.add(ti)
        assigned_dets.add(di)

    unmatched_tracks = [ti for ti in range(iou_matrix.shape[0])
                        if ti not in assigned_tracks]
    unmatched_dets = [di for di in range(iou_matrix.shape[1])
                      if di not in assigned_dets]

    return matched, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------

# A track becomes "confirmed" (eligible for motion analysis) after this many hits.
_MIN_HITS = 1


class IoUTracker:
    """
    Lightweight IoU-based multi-object tracker with per-track Kalman filtering.

    Usage
    -----
    tracker = IoUTracker(iou_threshold=0.30, max_age=5)

    # Each frame:
    tracked = tracker.update(detections, frame_index=n, timestamp=t)

    # tracked is a list of TrackedDetection objects — same detections
    # with a stable track_id added.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self._tracks: Dict[int, Track] = {}   # track_id → Track
        self._next_id: int = 1
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_index: int,
        timestamp: float,
    ) -> List[TrackedDetection]:
        """
        Process one frame of detections.

        Args:
            detections:  Raw Detection objects from YOLO (current frame).
            frame_index: Monotonically increasing frame counter.
            timestamp:   Unix timestamp (seconds) of this frame.

        Returns:
            List of TrackedDetection — each Detection paired with its
            stable track_id.  Only confirmed tracks are returned.
        """
        self._frame_count += 1
        active_tracks = list(self._tracks.values())

        # --- Predict all tracks forward ---
        for track in active_tracks:
            track.predict()

        # --- Match detections to tracks ---
        iou_mat = _build_iou_matrix(active_tracks, detections)
        matched, unmatched_track_idxs, unmatched_det_idxs = _greedy_match(
            iou_mat, self.iou_threshold
        )

        # --- Update matched tracks ---
        for ti, di in matched:
            det = detections[di]
            track = active_tracks[ti]
            meas = np.array(
                [det.center_x, det.center_y, det.width, det.height],
                dtype=np.float64,
            )
            track.update(meas)
            track.label = det.label
            if track.hits >= _MIN_HITS:
                track.is_confirmed = True

        # --- Create new tracks for unmatched detections ---
        for di in unmatched_det_idxs:
            det = detections[di]
            new_track = Track.from_detection(det, self._next_id)
            new_track.is_confirmed = True   # confirm immediately (1-hit confirm)
            self._tracks[self._next_id] = new_track
            logger.debug(f"New track created: id={self._next_id} label='{det.label}'")
            self._next_id += 1

        # --- Remove tracks that have been missing too long ---
        dead_ids = [
            tid for tid, track in self._tracks.items()
            if track.consecutive_misses > self.max_age
        ]
        for tid in dead_ids:
            logger.debug(f"Track expired: id={tid}")
            del self._tracks[tid]

        # --- Build output — only confirmed, currently matched tracks ---
        result: List[TrackedDetection] = []
        for ti, di in matched:
            track = active_tracks[ti]
            if track.is_confirmed:
                result.append(TrackedDetection(
                    detection=detections[di],
                    track_id=track.track_id,
                ))

        # Also include brand-new tracks (created in this frame)
        for di in unmatched_det_idxs:
            det = detections[di]
            # Find the track we just created for this detection
            for tid, track in self._tracks.items():
                if track.label == det.label and track.hits == 1 and track.consecutive_misses == 0:
                    # Match by position proximity (center within 10px)
                    tx, ty, tw, th = track.predicted_bbox
                    tcx = tx + tw // 2
                    tcy = ty + th // 2
                    if abs(tcx - det.center_x) <= 10 and abs(tcy - det.center_y) <= 10:
                        result.append(TrackedDetection(
                            detection=det,
                            track_id=tid,
                        ))
                        break

        logger.debug(
            f"Tracker frame {frame_index}: "
            f"{len(detections)} detections → {len(result)} tracked objects, "
            f"{len(self._tracks)} active tracks"
        )

        return result

    @property
    def active_track_ids(self) -> List[int]:
        """Returns IDs of all currently active (not-yet-expired) tracks."""
        return list(self._tracks.keys())

    def reset(self) -> None:
        """Clears all tracks and resets the ID counter."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("IoUTracker reset.")
