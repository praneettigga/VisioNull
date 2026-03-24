"""In-memory clip cache with TTL for transient pre-fall videos."""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CachedClip:
    clip_bytes: bytes
    mime_type: str
    created_at: float
    expires_at: float


class ClipCache:
    """Thread-safe, RAM-only clip store keyed by event row ID."""

    def __init__(self, ttl_seconds: int):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.ttl_seconds = ttl_seconds
        self._clips: Dict[int, CachedClip] = {}
        self._lock = threading.Lock()

    def put(self, event_id: int, clip_bytes: bytes, mime_type: str) -> None:
        now = time.time()
        with self._lock:
            self._purge_expired_locked(now)
            self._clips[event_id] = CachedClip(
                clip_bytes=clip_bytes,
                mime_type=mime_type,
                created_at=now,
                expires_at=now + self.ttl_seconds,
            )

    def get(self, event_id: int) -> Optional[CachedClip]:
        now = time.time()
        with self._lock:
            self._purge_expired_locked(now)
            return self._clips.get(event_id)

    def has_clip(self, event_id: int) -> bool:
        return self.get(event_id) is not None

    def _purge_expired_locked(self, now: float) -> None:
        expired_keys = [k for k, v in self._clips.items() if v.expires_at <= now]
        for key in expired_keys:
            del self._clips[key]


_clip_cache: Optional[ClipCache] = None


def init_clip_cache(ttl_seconds: int) -> ClipCache:
    global _clip_cache
    _clip_cache = ClipCache(ttl_seconds=ttl_seconds)
    return _clip_cache


def get_clip_cache() -> ClipCache:
    if _clip_cache is None:
        raise RuntimeError("Clip cache not initialized")
    return _clip_cache
