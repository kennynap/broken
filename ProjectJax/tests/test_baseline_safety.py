import json
from pathlib import Path

from jax.core import Jax, ToolResult
from jax.logging import OperationLogger
from jax.policy import Policy
from jax.tools import CreateDirectory, ListDirectory, ToolRegistry


class MutatingFakeTool:
    name = "mutating_fake"
    description = "A test mutation."
    mutates = True

    def spec(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, **kwargs):
        return ToolResult(True, {"changed": True})


class OneToolCallModel:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools, *, think=True):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "mutating_fake", "arguments": {}}}],
                }
            }
        return {"message": {"role": "assistant", "content": "finished"}}

    @staticmethod
    def encode_tool_result(result):
        return json.dumps({"ok": result.ok, "data": result.data, "error": result.error})


class RecordingLogger:
    def __init__(self):
        self.records = []

    def record(self, *args):
        self.records.append(args)


def test_mutation_requires_approval():
    logger = RecordingLogger()
    tool = MutatingFakeTool()
    jax = Jax(
        OneToolCallModel(),
        ToolRegistry([tool]),
        Policy(Path("/tmp"), input_fn=lambda _: "n"),
        logger,
    )

    result = jax.run("change something")

    assert result == "finished"
    assert logger.records
    assert logger.records[0][2].ok is False
    assert "did not approve" in logger.records[0][2].error


def test_operation_logger_writes_structured_record(tmp_path):
    log_path = tmp_path / "operations.jsonl"
    logger = OperationLogger(log_path)
    result = ToolResult(True, {"destination": "/tmp/example"})

    logger.record("move_file", {"source": "a", "destination": "b"}, result)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["tool"] == "move_file"
    assert record["ok"] is True
    assert record["data"]["destination"] == "/tmp/example"
    assert "timestamp" in record


def test_create_directory_fails_if_destination_exists(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    existing = root / "Documents"
    existing.mkdir()

    result = CreateDirectory(root).execute("Documents")

    assert result.ok is False
    assert existing.is_dir()


def test_path_escape_is_rejected_without_touching_outside(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    try:
        ListDirectory(root).execute(str(outside))
    except PermissionError:
        pass
    else:
        raise AssertionError("Path escape was not rejected")


def test_policy_allows_read_inside_root(tmp_path):
    policy = Policy(tmp_path)
    decision = policy.check(ListDirectory(tmp_path), {"path": "Documents"})
    assert decision.allowed


def test_policy_rejects_path_escape_before_tool_execution(tmp_path):
    policy = Policy(tmp_path)
    decision = policy.check(ListDirectory(tmp_path), {"path": "../outside"})
    assert not decision.allowed
    assert "outside authorized root" in decision.reason


def test_policy_rejects_absolute_path_outside_root(tmp_path):
    policy = Policy(tmp_path)
    outside = tmp_path.parent / "outside"
    decision = policy.check(ListDirectory(tmp_path), {"path": str(outside)})
    assert not decision.allowed


def test_policy_rejects_write_when_write_capability_missing(tmp_path):
    policy = Policy(tmp_path, capabilities={"filesystem.read"})
    decision = policy.check(CreateDirectory(tmp_path), {"path": "Documents"})
    assert not decision.allowed
    assert "filesystem.write" in decision.reason


def test_policy_allows_write_when_write_capability_present(tmp_path):
    policy = Policy(tmp_path)
    decision = policy.check(CreateDirectory(tmp_path), {"path": "Documents"})
    assert decision.allowed


def test_policy_resolves_symlink_escape(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)

    policy = Policy(root)
    decision = policy.check(ListDirectory(root), {"path": "escape"})
    assert not decision.allowed


def test_policy_does_not_create_or_modify_files(tmp_path):
    policy = Policy(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    policy.check(CreateDirectory(tmp_path), {"path": "Documents"})
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
