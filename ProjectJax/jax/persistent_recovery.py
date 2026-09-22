from __future__ import annotations

import json
from pathlib import Path

from .recovery import MoveRecord, RecoveryManager
from .verification import file_digest


class PersistentRecoveryManager(RecoveryManager):
    """Recovery journal that survives a JAX process restart."""

    def __init__(self, journal: Path):
        super().__init__()
        self.journal = journal.expanduser()
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def record_move(self, source: Path, destination: Path) -> MoveRecord:
        record = super().record_move(source, destination)
        self._append({
            "op": "record",
            "source": str(record.source),
            "destination": str(record.destination),
            "digest": record.digest,
        })
        return record

    def undo_last(self) -> MoveRecord:
        record = super().undo_last()
        self._append({
            "op": "undo",
            "source": str(record.source),
            "destination": str(record.destination),
            "digest": record.digest,
        })
        return record

    def _append(self, entry: dict) -> None:
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()

    def _load(self) -> None:
        if not self.journal.exists():
            return
        records: list[MoveRecord] = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            source = Path(entry["source"])
            destination = Path(entry["destination"])
            digest = entry["digest"]
            if entry["op"] == "record":
                records.append(MoveRecord(source, destination, digest))
            elif entry["op"] == "undo" and records:
                for index in range(len(records) - 1, -1, -1):
                    if records[index].source == source and records[index].destination == destination and records[index].digest == digest:
                        records.pop(index)
                        break
        self._moves.extend(records)
