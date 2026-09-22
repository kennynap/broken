from pathlib import Path

import pytest

from jax.model_validation import ModelResponseError, validate_chat_response
from jax.recovery import RecoveryManager
from jax.tasks import TaskManager, TaskStatus
from jax.verification import file_digest, verify_move, VerificationError


def test_verify_move_checks_content(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("hello")
    digest = file_digest(source)
    source.rename(destination)
    verify_move(source, destination, digest)


def test_verify_move_rejects_changed_content(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("hello")
    digest = file_digest(source)
    source.rename(destination)
    destination.write_text("changed")
    with pytest.raises(VerificationError):
        verify_move(source, destination, digest)


def test_recovery_undoes_verified_move(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "a.txt"
    destination = root / "Documents" / "a.txt"
    destination.parent.mkdir()
    source.write_text("important")
    source.rename(destination)
    recovery = RecoveryManager()
    recovery.record_move(source, destination)
    recovery.undo_last()
    assert source.read_text() == "important"
    assert not destination.exists()
    assert not recovery.can_undo()


def test_recovery_refuses_modified_destination(tmp_path):
    source = tmp_path / "a.txt"
    destination = tmp_path / "b.txt"
    source.write_text("original")
    source.rename(destination)
    recovery = RecoveryManager()
    recovery.record_move(source, destination)
    destination.write_text("tampered")
    with pytest.raises(RuntimeError):
        recovery.undo_last()
    assert destination.read_text() == "tampered"
    assert not source.exists()


def test_recovery_refuses_existing_source(tmp_path):
    source = tmp_path / "a.txt"
    destination = tmp_path / "b.txt"
    source.write_text("one")
    source.rename(destination)
    source.write_text("other")
    recovery = RecoveryManager()
    recovery.record_move(source, destination)
    with pytest.raises(FileExistsError):
        recovery.undo_last()


def test_task_lifecycle():
    manager = TaskManager()
    task = manager.create("organize downloads")
    assert task.status is TaskStatus.CREATED
    manager.start(task)
    manager.complete(task, "done")
    assert task.status is TaskStatus.COMPLETED
    assert task.result == "done"


def test_task_invalid_transition_rejected():
    manager = TaskManager()
    task = manager.create("test")
    with pytest.raises(ValueError):
        manager.complete(task, "done")


def test_model_response_validation():
    message = validate_chat_response({"message": {"content": "ok", "tool_calls": []}})
    assert message["content"] == "ok"


def test_model_response_rejects_malformed_tool_call():
    with pytest.raises(ModelResponseError):
        validate_chat_response({"message": {"tool_calls": [{"function": {"name": "x", "arguments": []}}]}})


def test_task_failure_records_error():
    manager = TaskManager()
    task = manager.create("fail")
    manager.start(task)
    manager.fail(task, "boom")
    assert task.status is TaskStatus.FAILED
    assert task.error == "boom"
