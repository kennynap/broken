from pathlib import Path
import pytest

from jax.organizer import FileClassifier, OrganizationPlanner, OrganizationExecutor, default_rules
from jax.persistent_recovery import PersistentRecoveryManager
from jax.recovery import RecoveryManager
from jax.transactions import OrganizationTransaction


def make_plan(root):
    (root / "a.pdf").write_text("a")
    (root / "b.jpg").write_text("b")
    return OrganizationPlanner(FileClassifier(default_rules())).plan(root)


def test_dry_run_does_not_modify_files(tmp_path):
    plan = make_plan(tmp_path)
    completed = OrganizationExecutor().execute(plan, dry_run=True)
    assert len(completed) == 2
    assert (tmp_path / "a.pdf").exists()
    assert (tmp_path / "b.jpg").exists()
    assert not (tmp_path / "Documents" / "a.pdf").exists()


def test_transaction_rolls_back_completed_moves_on_failure(tmp_path):
    plan = make_plan(tmp_path)
    recovery = RecoveryManager()
    result = OrganizationTransaction(recovery).run(plan, fail_after=1, atomic=True)
    assert result.rolled_back is True
    assert (tmp_path / "a.pdf").exists()
    assert (tmp_path / "b.jpg").exists()
    assert not (tmp_path / "Documents" / "a.pdf").exists()
    assert not (tmp_path / "Pictures" / "b.jpg").exists()
    assert not recovery.can_undo()


def test_transaction_rejects_duplicate_destinations(tmp_path):
    plan = make_plan(tmp_path)
    duplicate = plan.actions[0]
    bad = plan.__class__(plan.root, (duplicate, duplicate), plan.skipped)
    with pytest.raises(ValueError, match="Duplicate destination"):
        OrganizationTransaction(RecoveryManager()).run(bad)


def test_persistent_recovery_survives_restart(tmp_path):
    source = tmp_path / "a.txt"
    destination = tmp_path / "Documents" / "a.txt"
    source.write_text("persistent")
    destination.parent.mkdir()
    source.rename(destination)

    journal = tmp_path / "recovery.jsonl"
    first = PersistentRecoveryManager(journal)
    first.record_move(source, destination)

    second = PersistentRecoveryManager(journal)
    assert second.can_undo()
    second.undo_last()
    assert source.read_text() == "persistent"
    assert not destination.exists()


def test_persistent_recovery_records_undo(tmp_path):
    source = tmp_path / "a.txt"
    destination = tmp_path / "b.txt"
    source.write_text("x")
    source.rename(destination)
    journal = tmp_path / "recovery.jsonl"
    recovery = PersistentRecoveryManager(journal)
    recovery.record_move(source, destination)
    recovery.undo_last()
    restarted = PersistentRecoveryManager(journal)
    assert not restarted.can_undo()
