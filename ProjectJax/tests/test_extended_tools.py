import os
import sys
from pathlib import Path

import pytest

from jax.builtins import filesystem_tools
from jax.executor import ExecutionRequest, ToolExecutor
from jax.logging import OperationLogger
from jax.policy import Policy
from jax.process import ProcessError, ProcessPolicy, ProcessRunner, linux_processes
from jax.result import ToolResult
from jax.schema import validate_tool_arguments
from jax.session import SessionStore
from jax.tools import ToolRegistry


def executor(tmp_path, tools):
    return ToolExecutor(ToolRegistry(tools), Policy(tmp_path, input_fn=lambda _: "y"), OperationLogger(tmp_path / "ops.jsonl"))


def test_read_search_and_write_tools(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    ex = executor(tmp_path, filesystem_tools(tmp_path))
    found = ex.execute(ExecutionRequest("search_files", {"pattern": "*.txt"}))
    assert found.ok and found.data["matches"][0]["path"].endswith("a.txt")
    read = ex.execute(ExecutionRequest("read_text_file", {"path": "a.txt"}))
    assert read.ok and read.data["text"] == "hello"
    written = ex.execute(ExecutionRequest("write_text_file", {"path": "new/b.txt", "text": "world"}))
    assert written.ok and (tmp_path / "new/b.txt").read_text() == "world"
    blocked = ex.execute(ExecutionRequest("write_text_file", {"path": "new/b.txt", "text": "overwrite"}))
    assert not blocked.ok


def test_search_does_not_follow_symlinks(tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    (tmp_path / "link").symlink_to(outside)
    ex = executor(tmp_path, filesystem_tools(tmp_path))
    result = ex.execute(ExecutionRequest("search_files", {"pattern": "*"}))
    assert result.ok
    assert all(not item["path"].endswith("secret.txt") for item in result.data["matches"])


def test_system_info_and_disk_usage(tmp_path):
    ex = executor(tmp_path, filesystem_tools(tmp_path))
    info = ex.execute(ExecutionRequest("system_info", {}))
    usage = ex.execute(ExecutionRequest("disk_usage", {"path": "."}))
    assert info.ok and info.data["python"]
    assert usage.ok and usage.data["total"] >= usage.data["free"]


def test_process_policy_blocks_shell_and_unapproved_commands():
    runner = ProcessRunner(ProcessPolicy(frozenset({Path(sys.executable).name})))
    result = runner.run([sys.executable, "-c", "print('ok')"])
    assert result.returncode == 0 and result.stdout.strip() == "ok"
    with pytest.raises(ProcessError):
        runner.run(["sh", "-c", "echo bad"])
    with pytest.raises(ProcessError):
        runner.run([sys.executable, "-c", "print('bad')"], timeout=99)


def test_process_runner_has_no_shell_interpretation():
    runner = ProcessRunner(ProcessPolicy(frozenset({Path(sys.executable).name})))
    result = runner.run([sys.executable, "-c", "print('a;b')"])
    assert result.returncode == 0 and result.stdout.strip() == "a;b"


def test_process_timeout_is_reported():
    runner = ProcessRunner(ProcessPolicy(frozenset({Path(sys.executable).name}), max_timeout=2))
    result = runner.run([sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.1)
    assert result.timed_out is True


def test_linux_process_inventory_contains_current_process():
    if not Path("/proc").is_dir():
        pytest.skip("Linux /proc unavailable")
    processes = linux_processes()
    assert any(item["pid"] == os.getpid() for item in processes)


def test_session_store_persists_and_bounds_history(tmp_path):
    store = SessionStore(tmp_path, max_messages=3)
    session = store.create()
    for index in range(5):
        store.append(session, "user", str(index))
    loaded = store.load(session)
    assert [m.content for m in loaded] == ["2", "3", "4"]
    reopened = SessionStore(tmp_path, max_messages=3)
    assert [m.content for m in reopened.load(session)] == ["2", "3", "4"]


def test_session_rejects_path_traversal_and_invalid_role(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.load("../escape")
    session = store.create()
    with pytest.raises(ValueError):
        store.append(session, "root", "bad")


def test_plan_preflight_rejects_entire_invalid_batch_without_mutation(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    ex = executor(tmp_path, filesystem_tools(tmp_path))
    from jax.planning import PlannedCall, ToolPlan
    plan = ToolPlan([
        PlannedCall("create_directory", {"path": "Dest"}),
        PlannedCall("move_file", {"source": "a.txt", "destination": "../escape.txt"}),
    ])
    check = plan.preflight(ex)
    assert not check.allowed
    assert (tmp_path / "Dest").exists() is False
    assert (tmp_path / "a.txt").exists()

def test_filesystem_builtin_bundle_has_unique_names(tmp_path):
    tools = filesystem_tools(tmp_path)
    names = [tool.name for tool in tools]
    assert len(names) == len(set(names))
    assert {"list_directory", "search_files", "read_text_file", "write_text_file", "system_info"}.issubset(names)

def test_process_list_tool_is_authorized_and_returns_processes(tmp_path):
    from jax.process_tools import ListProcesses
    ex = executor(tmp_path, [ListProcesses()])
    result = ex.execute(ExecutionRequest("list_processes", {"limit": 10}))
    assert result.ok
    assert isinstance(result.data["processes"], list)


def test_jax_can_persist_conversation_history(tmp_path):
    import json
    from jax.core import Jax
    from jax.events import EventBus
    from jax.tasks import TaskManager
    from jax.model import OllamaModel

    class Model:
        def chat(self, messages, tools, *, think=True):
            return {"message": {"role": "assistant", "content": "stored response"}}
        @staticmethod
        def encode_tool_result(result):
            return json.dumps({"ok": result.ok})

    from jax.session import SessionStore
    from jax.logging import OperationLogger
    store = SessionStore(tmp_path / "sessions")
    jax = Jax(Model(), ToolRegistry([]), Policy(tmp_path), OperationLogger(tmp_path / "ops.jsonl"), session_store=store)
    assert jax.run("remember this") == "stored response"
    messages = store.load(jax.session_id)
    assert [(m.role, m.content) for m in messages] == [("user", "remember this"), ("assistant", "stored response")]


def test_production_registry_exposes_agent_capabilities(tmp_path, monkeypatch):
    from jax.main import build_jax

    monkeypatch.setenv("JAX_ALLOWED_ROOT", str(tmp_path))
    jax = build_jax()
    names = {tool.name for tool in jax.registry._tools.values()}

    assert {"list_directory", "search_files", "read_text_file", "write_text_file", "move_file", "run_command"}.issubset(names)
    assert "list_processes" in names
    assert "system_info" in names


def test_controlled_filesystem_write_move_and_outside_boundary(tmp_path):
    ex = executor(tmp_path, filesystem_tools(tmp_path))
    written = ex.execute(ExecutionRequest(
        "write_text_file",
        {"path": "Inbox/hello.py", "text": 'print("Hello, World!")\n'},
    ))
    assert written.ok
    assert (tmp_path / "Inbox/hello.py").read_text() == 'print("Hello, World!")\n'

    moved = ex.execute(ExecutionRequest(
        "move_file",
        {"source": "Inbox/hello.py", "destination": "Sorted/hello.py"},
    ))
    assert moved.ok
    assert not (tmp_path / "Inbox/hello.py").exists()
    assert (tmp_path / "Sorted/hello.py").is_file()

    blocked = ex.execute(ExecutionRequest(
        "move_file",
        {"source": "Sorted/hello.py", "destination": "../outside.py"},
    ))
    assert not blocked.ok
    assert (tmp_path / "Sorted/hello.py").is_file()
