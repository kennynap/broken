from jax.core import ToolResult
from jax.executor import ExecutionRequest, ToolExecutor
from jax.tools import BaseTool, ToolRegistry


class FakeTool(BaseTool):
    name = "fake"
    description = "fake"
    mutates = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return ToolResult(True, {"ok": True})


class BadTool(FakeTool):
    name = "bad"
    def execute(self, **kwargs):
        raise RuntimeError("boom")


class FakePolicy:
    def check(self, tool, args):
        return type("Decision", (), {"allowed": True, "reason": "allowed"})()
    def confirm(self, tool, args):
        return True


class FakeLogger:
    def __init__(self):
        self.records = []
    def record(self, *args):
        self.records.append(args)


def test_executor_runs_tool_and_logs():
    logger = FakeLogger()
    executor = ToolExecutor(ToolRegistry([FakeTool()]), FakePolicy(), logger)
    result = executor.execute(ExecutionRequest("fake", {}))
    assert result.ok
    assert len(logger.records) == 1


def test_executor_rejects_unknown_tool():
    logger = FakeLogger()
    executor = ToolExecutor(ToolRegistry([]), FakePolicy(), logger)
    result = executor.execute(ExecutionRequest("missing", {}))
    assert not result.ok
    assert "Unknown tool" in result.error


def test_executor_converts_tool_exception():
    logger = FakeLogger()
    executor = ToolExecutor(ToolRegistry([BadTool()]), FakePolicy(), logger)
    result = executor.execute(ExecutionRequest("bad", {}))
    assert not result.ok
    assert "RuntimeError" in result.error


def test_executor_rejects_non_object_arguments():
    logger = FakeLogger()
    executor = ToolExecutor(ToolRegistry([FakeTool()]), FakePolicy(), logger)
    result = executor.execute(ExecutionRequest("fake", []))
    assert not result.ok
    assert "arguments must be an object" in result.error
