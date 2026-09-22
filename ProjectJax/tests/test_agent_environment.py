import os
import shutil

import pytest

from jax.builtins import filesystem_tools
from jax.executor import ExecutionRequest, ToolExecutor
from jax.logging import OperationLogger
from jax.policy import Policy
from jax.process_tools import RunCommand
from jax.tools import ToolRegistry


def test_production_agent_exposes_run_command(tmp_path):
    names = {tool.name for tool in filesystem_tools(tmp_path)}
    assert "run_command" in names


def test_run_command_requires_bubblewrap_and_never_falls_back(tmp_path):
    tool = RunCommand(tmp_path)
    result = tool.execute("printf 'hello\\n'")
    if shutil.which("bwrap") or shutil.which("bubblewrap"):
        assert result.ok
        assert result.data["returncode"] == 0
        assert result.data["stdout"] == "hello\\n"
    else:
        assert not result.ok
        assert "bubblewrap" in result.error


@pytest.mark.skipif(not (shutil.which("bwrap") or shutil.which("bubblewrap")), reason="bubblewrap not installed")
def test_run_command_can_reason_about_workspace_state(tmp_path):
    tool = RunCommand(tmp_path)
    result = tool.execute("mkdir -p project && printf 'hello' > project/a.txt && cat project/a.txt")
    assert result.ok
    assert result.data["returncode"] == 0
    assert result.data["stdout"] == "hello"
    assert (tmp_path / "project" / "a.txt").read_text() == "hello"


@pytest.mark.skipif(not (shutil.which("bwrap") or shutil.which("bubblewrap")), reason="bubblewrap not installed")
def test_run_command_cannot_write_outside_workspace(tmp_path):
    outside = tmp_path.parent / "jax-outside-test.txt"
    if outside.exists():
        outside.unlink()
    tool = RunCommand(tmp_path)
    result = tool.execute(f"printf blocked > '{outside}'")
    assert result.ok
    assert result.data["returncode"] != 0 or not outside.exists()
    assert not outside.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="JAX must not run as root")
def test_policy_default_is_autonomous(tmp_path):
    from jax.tools import CreateDirectory
    ex = ToolExecutor(
        ToolRegistry([CreateDirectory(tmp_path)]),
        Policy(tmp_path),
        OperationLogger(tmp_path / "ops.jsonl"),
    )
    result = ex.execute(ExecutionRequest("create_directory", {"path": "new"}))
    assert result.ok
    assert (tmp_path / "new").is_dir()
