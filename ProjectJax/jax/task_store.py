from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from .tasks import Task, TaskManager, TaskStatus


class PersistentTaskManager(TaskManager):
    """Task manager whose state survives a process restart."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Task store must contain a list")
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Invalid task record")
            task = Task(
                request=item["request"], id=item["id"],
                status=TaskStatus(item["status"]),
                result=item.get("result"), error=item.get("error"),
            )
            self.tasks[task.id] = task

    def _save(self) -> None:
        payload = [
            {"id": t.id, "request": t.request, "status": t.status.value,
             "result": t.result, "error": t.error}
            for t in self.tasks.values()
        ]
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)

    def create(self, request: str) -> Task:
        with self._lock:
            task = super().create(request)
            self._save()
            return task

    def start(self, task: Task) -> None:
        with self._lock:
            super().start(task)
            self._save()

    def complete(self, task: Task, result: str) -> None:
        with self._lock:
            super().complete(task, result)
            self._save()

    def fail(self, task: Task, error: str) -> None:
        with self._lock:
            super().fail(task, error)
            self._save()

    def recover_interrupted(self) -> list[Task]:
        """Mark tasks left running by a previous process as failed."""
        with self._lock:
            interrupted = [t for t in self.tasks.values() if t.status is TaskStatus.RUNNING]
            for task in interrupted:
                task.status = TaskStatus.FAILED
                task.error = "Task interrupted by process restart"
            if interrupted:
                self._save()
            return interrupted
