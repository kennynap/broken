from pathlib import Path

import pytest

from jax.schema import ToolArgumentError, validate_tool_arguments
from jax.task_store import PersistentTaskManager
from jax.tasks import TaskStatus
from jax.tools import ListDirectory


def test_persistent_tasks_survive_restart(tmp_path):
    path = tmp_path / "tasks.json"
    first = PersistentTaskManager(path)
    task = first.create("organize downloads")
    first.start(task)

    second = PersistentTaskManager(path)
    loaded = second.tasks[task.id]
    assert loaded.status is TaskStatus.RUNNING
    interrupted = second.recover_interrupted()
    assert interrupted[0].id == task.id

    third = PersistentTaskManager(path)
    assert third.tasks[task.id].status is TaskStatus.FAILED


def test_task_store_rejects_malformed_json(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        PersistentTaskManager(path)


def test_schema_accepts_valid_tool_arguments(tmp_path):
    tool = ListDirectory(tmp_path)
    validate_tool_arguments(tool, {"path": "."})


def test_schema_rejects_unknown_argument(tmp_path):
    tool = ListDirectory(tmp_path)
    with pytest.raises(ToolArgumentError):
        validate_tool_arguments(tool, {"path": ".", "extra": "bad"})


def test_schema_rejects_wrong_type(tmp_path):
    tool = ListDirectory(tmp_path)
    with pytest.raises(ToolArgumentError):
        validate_tool_arguments(tool, {"path": 7})
