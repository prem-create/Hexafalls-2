"""
Walking Eye - AI Perception Engine
Temporal History Buffer.

Maintains a bounded per-object ring buffer of recent observations.
Each observation is a snapshot of everything known about a tracked object
at a specific point in time: bounding-box geometry, timestamp, frame index,
and optional metric depth.

Design choices
--------------
* Uses collections.deque(maxlen=N) — O(1) append/pop, bounded memory.
* One TemporalBuffer instance per tracked object; the TrackerStore owns
  a mapping of track_id → TemporalBuffer.
* Depth is Optional[float].  When unavailable, motion analysis falls back
  to bounding-box scale heuristics and never fabricates metric values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass(frozen=True)
class Observation:
    """
    A single frame-level snapshot of a tracked object.

    All pixel coordinates are in the *processed* (possibly resized) image space.
    Depth, if present, is in metres (positive = distance in front of camera).
    """

    track_id: int
    timestamp: float           # Unix time in seconds (float for sub-second precision)
    frame_index: int

    # Bounding box (pixel coordinates, top-left origin)
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int

    # Derived convenience fields (stored to avoid recomputing)
    center_x: int
    center_y: int
    bbox_area: int             # width * height

    confidence: float

    # Depth in metres.  None when depth estimation is unavailable.
    estimated_depth: Optional[float] = None

    @classmethod
    def from_values(
        cls,
        *,
        track_id: int,
        timestamp: float,
        frame_index: int,
        bbox_x: int,
        bbox_y: int,
        bbox_width: int,
        bbox_height: int,
        confidence: float,
        estimated_depth: Optional[float] = None,
    ) -> "Observation":
        """Convenience constructor that computes derived fields automatically."""
        return cls(
            track_id=track_id,
            timestamp=timestamp,
            frame_index=frame_index,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            center_x=bbox_x + bbox_width // 2,
            center_y=bbox_y + bbox_height // 2,
            bbox_area=bbox_width * bbox_height,
            confidence=confidence,
            estimated_depth=estimated_depth,
        )


class TemporalBuffer:
    """
    Fixed-size ring buffer storing the N most recent Observations for one track.

    Oldest entries are automatically discarded when the buffer is full.
    """

    def __init__(self, history_size: int = 5) -> None:
        """
        Args:
            history_size: Maximum number of observations to retain.
                          Must be >= 1.  Recommended: 3–10.
        """
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        self._history_size = history_size
        self._buffer: Deque[Observation] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, observation: Observation) -> None:
        """Add a new observation.  The oldest is dropped if the buffer is full."""
        self._buffer.append(observation)

    def clear(self) -> None:
        """Remove all stored observations (e.g. when a track is reset)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of observations currently stored."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def latest(self) -> Optional[Observation]:
        """Most recent observation, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    @property
    def oldest(self) -> Optional[Observation]:
        """Oldest observation currently in the buffer, or None if empty."""
        return self._buffer[0] if self._buffer else None

    def as_list(self) -> List[Observation]:
        """Return all observations, oldest first."""
        return list(self._buffer)

    def recent(self, n: int) -> List[Observation]:
        """
        Return the most-recent `n` observations, oldest first within the slice.

        If fewer than `n` are available, all stored observations are returned.
        """
        items = list(self._buffer)
        return items[-n:] if len(items) >= n else items

    # ------------------------------------------------------------------
    # Convenience analysis helpers
    # ------------------------------------------------------------------

    def depth_sequence(self) -> List[Optional[float]]:
        """Returns the depth values in chronological order (oldest first)."""
        return [obs.estimated_depth for obs in self._buffer]

    def bbox_height_sequence(self) -> List[int]:
        """Returns bbox heights in chronological order."""
        return [obs.bbox_height for obs in self._buffer]

    def bbox_area_sequence(self) -> List[int]:
        """Returns bbox areas in chronological order."""
        return [obs.bbox_area for obs in self._buffer]

    def center_sequence(self) -> List[tuple]:
        """Returns (center_x, center_y) tuples in chronological order."""
        return [(obs.center_x, obs.center_y) for obs in self._buffer]

    def time_span_seconds(self) -> float:
        """
        Wall-clock duration from oldest to latest observation in the buffer.
        Returns 0.0 if fewer than 2 observations are stored.
        """
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp

    def __repr__(self) -> str:
        latest_ts = self._buffer[-1].timestamp if self._buffer else None
        return (
            f"TemporalBuffer(count={self.count}/{self._history_size}, "
            f"latest_ts={latest_ts})"
        )
