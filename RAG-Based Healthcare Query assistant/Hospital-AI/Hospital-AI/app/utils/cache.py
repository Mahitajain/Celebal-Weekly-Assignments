"""
Lightweight in-memory TTL + LRU cache for chat responses.

Repeated identical questions (common in a demo/interview setting, and in
real usage -- "how many diabetic patients" gets asked a lot) shouldn't
re-run an LLM call and a database query every time. This is a thread-safe
`OrderedDict`-based cache with both a max size (LRU eviction) and a TTL
(so stale answers don't linger if the underlying data changes). For a
multi-instance deployment this would move to Redis with the same
`get`/`set` interface.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import get_settings


class TTLCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            timestamp, value = entry
            if time.time() - timestamp > self.ttl_seconds:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_cache_singleton: TTLCache | None = None


def get_query_cache() -> TTLCache:
    global _cache_singleton
    if _cache_singleton is None:
        settings = get_settings()
        _cache_singleton = TTLCache(settings.query_cache_max_size, settings.query_cache_ttl_seconds)
    return _cache_singleton
