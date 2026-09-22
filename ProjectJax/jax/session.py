from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    timestamp: str


class SessionStore:
    """Small persistent conversation store with bounded history."""

    ROLES = frozenset({"system", "user", "assistant", "tool"})

    def __init__(self, directory: Path, max_messages: int = 100):
        if max_messages < 2 or max_messages > 10000:
            raise ValueError("max_messages must be between 2 and 10000")
        self.directory = directory.expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_messages = max_messages
        self._lock = RLock()

    def create(self) -> str:
        session_id = uuid4().hex
        self._path(session_id).write_text("[]", encoding="utf-8")
        return session_id

    def append(self, session_id: str, role: str, content: str) -> Message:
        if role not in self.ROLES:
            raise ValueError(f"Invalid message role: {role}")
        if not isinstance(content, str):
            raise TypeError("Message content must be text")
        with self._lock:
            messages = self.load(session_id)
            message = Message(role, content, datetime.now(timezone.utc).isoformat())
            messages.append(message)
            messages = messages[-self.max_messages:]
            self._atomic_write(session_id, messages)
            return message

    def load(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError(session_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Session file must contain a list")
        return [Message(item["role"], item["content"], item["timestamp"]) for item in raw]

    def clear(self, session_id: str) -> None:
        self._path(session_id).write_text("[]", encoding="utf-8")

    def _path(self, session_id: str) -> Path:
        if not session_id or Path(session_id).name != session_id or session_id != session_id.strip():
            raise ValueError("Invalid session id")
        return self.directory / f"{session_id}.json"

    def _atomic_write(self, session_id: str, messages: list[Message]) -> None:
        target = self._path(session_id)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps([asdict(m) for m in messages], indent=2), encoding="utf-8")
        temp.replace(target)
