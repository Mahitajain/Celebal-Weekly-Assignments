"""
Conversation memory.

Bounded, in-memory, per-session turn history. A real multi-instance
production deployment would back this with Redis or a database row
keyed by session id; the interface here (`get`, `append`, `clear`) is
intentionally the same shape that swap would need, so it's a drop-in
change rather than a rewrite. In-memory is the right choice for a
single-process portfolio deployment: it's zero-setup and the state loss
on restart is an acceptable trade-off for a chat session.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque

from app.config import get_settings
from app.orchestrator.state import ConversationTurn


class ConversationMemory:
    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns or get_settings().conversation_memory_turns
        self._lock = threading.Lock()
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_turns * 2))

    def get(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            return list(self._sessions[session_id])

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._sessions[session_id].append(ConversationTurn(role=role, content=content))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


_memory_singleton: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory_singleton
    if _memory_singleton is None:
        _memory_singleton = ConversationMemory()
    return _memory_singleton
