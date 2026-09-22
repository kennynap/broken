from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    request: str
    id: str = field(default_factory=lambda: uuid4().hex)
    status: TaskStatus = TaskStatus.CREATED
    result: str | None = None
    error: str | None = None


class TaskManager:
    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def create(self, request: str) -> Task:
        task = Task(request=request)
        self.tasks[task.id] = task
        return task

    def start(self, task: Task) -> None:
        if task.status is not TaskStatus.CREATED:
            raise ValueError("Only a newly created task can be started")
        task.status = TaskStatus.RUNNING

    def complete(self, task: Task, result: str) -> None:
        if task.status is not TaskStatus.RUNNING:
            raise ValueError("Only a running task can be completed")
        task.status = TaskStatus.COMPLETED
        task.result = result

    def fail(self, task: Task, error: str) -> None:
        if task.status is not TaskStatus.RUNNING:
            raise ValueError("Only a running task can fail")
        task.status = TaskStatus.FAILED
        task.error = error
