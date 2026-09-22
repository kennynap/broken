import json
from pathlib import Path

import pytest

from jax.config import Config
from jax.core import Jax
from jax.events import EventBus
from jax.logging import OperationLogger
from jax.policy import Policy
from jax.task_store import PersistentTaskManager
from jax.tools import ToolRegistry


class FinalModel:
    def chat(self, messages, tools, *, think=True):
        return {"message": {"role": "assistant", "content": "done"}}

    @staticmethod
    def encode_tool_result(result):
        return json.dumps({"ok": result.ok})


def test_persistent_tasks_and_events_survive_jax_run(tmp_path):
    events = []
    bus = EventBus()
    bus.subscribe("*", events.append)
    tasks = PersistentTaskManager(tmp_path / "tasks.json")
    jax = Jax(FinalModel(), ToolRegistry([]), Policy(tmp_path), OperationLogger(tmp_path / "ops.jsonl"), task_manager=tasks, event_bus=bus)
    assert jax.run("hello") == "done"
    assert [event.name for event in events] == ["task.started", "task.completed"]
    assert next(iter(tasks.tasks.values())).status.value == "completed"


def test_config_rejects_unsafe_root():
    config = Config(allowed_root=Path("/"))
    with pytest.raises(ValueError, match="filesystem root"):
        config.validate()


def test_config_rejects_bad_max_turns(monkeypatch):
    config = Config(max_turns=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        config.validate()
