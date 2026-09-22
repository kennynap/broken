from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .verification import file_digest, verify_move


@dataclass(frozen=True)
class MoveRecord:
    source: Path
    destination: Path
    digest: str


class RecoveryManager:
    """Tracks reversible filesystem moves and verifies undo results."""

    def __init__(self):
        self._moves: list[MoveRecord] = []

    def record_move(self, source: Path, destination: Path) -> MoveRecord:
        if not destination.is_file():
            raise FileNotFoundError(destination)
        record = MoveRecord(source.resolve(), destination.resolve(), file_digest(destination))
        self._moves.append(record)
        return record

    def undo_last(self) -> MoveRecord:
        if not self._moves:
            raise RuntimeError("There is no operation available to undo")
        record = self._moves[-1]
        if not record.destination.is_file():
            raise FileNotFoundError(record.destination)
        if record.source.exists():
            raise FileExistsError(record.source)
        if file_digest(record.destination) != record.digest:
            raise RuntimeError("Undo refused: destination content has changed")
        record.source.parent.mkdir(parents=True, exist_ok=True)
        record.destination.rename(record.source)
        verify_move(record.destination, record.source, record.digest)
        self._moves.pop()
        return record

    def can_undo(self) -> bool:
        return bool(self._moves)
