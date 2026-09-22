from jax.organizer import FileClassifier, OrganizationPlanner, OrganizationExecutor, default_rules
from jax.recovery import RecoveryManager
from jax.policy import Policy
from jax.tools import ListDirectory, MoveFile, ToolRegistry
from jax.executor import ExecutionRequest, ToolExecutor


class Logger:
    def __init__(self):
        self.records = []
    def record(self, *args):
        self.records.append(args)


def test_end_to_end_filesystem_flow(tmp_path):
    (tmp_path / "invoice.pdf").write_text("invoice")
    (tmp_path / "photo.jpg").write_text("photo")

    plan = OrganizationPlanner(FileClassifier(default_rules())).plan(tmp_path)
    recovery = RecoveryManager()
    completed = OrganizationExecutor().execute(plan, recovery)

    assert len(completed) == 2
    assert (tmp_path / "Documents" / "invoice.pdf").read_text() == "invoice"
    assert (tmp_path / "Pictures" / "photo.jpg").read_text() == "photo"

    recovery.undo_last()
    recovery.undo_last()
    assert (tmp_path / "invoice.pdf").exists()
    assert (tmp_path / "photo.jpg").exists()


def test_end_to_end_tool_policy_executor(tmp_path):
    root = tmp_path / "Downloads"
    root.mkdir()
    (root / "a.txt").write_text("a")
    logger = Logger()
    registry = ToolRegistry([ListDirectory(root), MoveFile(root)])
    executor = ToolExecutor(registry, Policy(root), logger)

    result = executor.execute(ExecutionRequest("list_directory", {"path": "."}))
    assert result.ok
    assert result.data["items"][0]["name"] == "a.txt"

    denied = executor.execute(ExecutionRequest("list_directory", {"path": "../outside"}))
    assert not denied.ok
    assert len(logger.records) == 2
