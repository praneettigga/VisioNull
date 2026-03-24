"""
In-memory rolling frame buffer for pre-fall clip extraction.

This module never writes frames to disk.
"""

import threading
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class PreFallFrameBuffer:
    """Time-based rolling frame buffer for the last N seconds."""

    def __init__(self, window_seconds: float, max_fps: int):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if max_fps <= 0:
            raise ValueError("max_fps must be > 0")

        self.window_seconds = float(window_seconds)
        # Safety headroom for clock jitter and FPS spikes.
        self._maxlen = int(window_seconds * max_fps * 2)
        self._frames: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    def add_frame(self, timestamp: float, frame: np.ndarray) -> None:
        """Append a frame and evict old frames outside the rolling window."""
        with self._lock:
            self._frames.append((timestamp, frame.copy()))
            self._evict_older_than(timestamp - self.window_seconds)

    def get_window(self, start_ts: float, end_ts: float) -> List[np.ndarray]:
        """Return frames where start_ts <= timestamp <= end_ts."""
        if end_ts < start_ts:
            return []
        with self._lock:
            return [frame for ts, frame in self._frames if start_ts <= ts <= end_ts]

    def get_last_seconds_until(self, end_ts: float, seconds: float) -> List[np.ndarray]:
        """Return up to `seconds` of frames ending at `end_ts`."""
        if seconds <= 0:
            return []
        return self.get_window(end_ts - seconds, end_ts)

    def latest_timestamp(self) -> Optional[float]:
        """Get most recent frame timestamp, if any."""
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1][0]

    def size(self) -> int:
        """Current number of buffered frames."""
        with self._lock:
            return len(self._frames)

    def clear(self) -> None:
        """Clear all buffered frames."""
        with self._lock:
            self._frames.clear()

    def _evict_older_than(self, min_ts: float) -> None:
        while self._frames and self._frames[0][0] < min_ts:
            self._frames.popleft()
