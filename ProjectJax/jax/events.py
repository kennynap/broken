from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable[[Event], None]]] = {}

    def subscribe(self, name: str, handler: Callable[[Event], None]) -> None:
        if not name:
            raise ValueError("Event name is required")
        self._handlers.setdefault(name, []).append(handler)

    def publish(self, name: str, **data: Any) -> Event:
        event = Event(name=name, data=data)
        for handler in tuple(self._handlers.get(name, ())):
            handler(event)
        for handler in tuple(self._handlers.get("*", ())):
            handler(event)
        return event

    def unsubscribe(self, name: str, handler: Callable[[Event], None]) -> None:
        handlers = self._handlers.get(name, [])
        if handler in handlers:
            handlers.remove(handler)
