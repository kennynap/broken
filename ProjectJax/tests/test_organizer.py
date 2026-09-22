from pathlib import Path

import pytest

from jax.organizer import (
    FileClassifier,
    OrganizationExecutor,
    OrganizationPlanner,
    default_rules,
)


def planner():
    return OrganizationPlanner(FileClassifier(default_rules()))


def test_plan_classifies_known_files_without_mutating(tmp_path):
    (tmp_path / "invoice.pdf").write_text("invoice")
    (tmp_path / "photo.jpg").write_text("photo")
    (tmp_path / "project.zip").write_text("zip")

    plan = planner().plan(tmp_path)

    assert [(a.source.name, a.destination.relative_to(tmp_path).as_posix()) for a in plan.actions] == [
        ("invoice.pdf", "Documents/invoice.pdf"),
        ("photo.jpg", "Pictures/photo.jpg"),
        ("project.zip", "Projects/project.zip"),
    ]
    assert not (tmp_path / "Documents").exists()
    assert not (tmp_path / "Pictures").exists()


def test_unknown_file_is_skipped(tmp_path):
    (tmp_path / "mystery.xyz").write_text("unknown")
    plan = planner().plan(tmp_path)
    assert not plan.actions
    assert len(plan.skipped) == 1
    assert plan.skipped[0].confidence == "unknown"


def test_existing_destination_is_skipped_without_overwrite(tmp_path):
    (tmp_path / "invoice.pdf").write_text("new")
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Documents" / "invoice.pdf").write_text("old")

    plan = planner().plan(tmp_path)

    assert not plan.actions
    assert plan.skipped[0].confidence == "blocked"
    assert (tmp_path / "invoice.pdf").read_text() == "new"
    assert (tmp_path / "Documents" / "invoice.pdf").read_text() == "old"


def test_executor_moves_and_verifies_files(tmp_path):
    (tmp_path / "invoice.pdf").write_text("invoice")
    plan = planner().plan(tmp_path)

    completed = OrganizationExecutor().execute(plan)

    assert len(completed) == 1
    assert not (tmp_path / "invoice.pdf").exists()
    assert (tmp_path / "Documents" / "invoice.pdf").read_text() == "invoice"


def test_executor_rejects_destination_created_after_planning(tmp_path):
    (tmp_path / "invoice.pdf").write_text("new")
    plan = planner().plan(tmp_path)
    destination = tmp_path / "Documents" / "invoice.pdf"
    destination.parent.mkdir()
    destination.write_text("existing")

    with pytest.raises(FileExistsError):
        OrganizationExecutor().execute(plan)

    assert (tmp_path / "invoice.pdf").exists()
    assert destination.read_text() == "existing"


def test_executor_rejects_action_outside_root(tmp_path):
    source = tmp_path / "invoice.pdf"
    source.write_text("invoice")
    outside = tmp_path.parent / "jax-outside-test.pdf"
    action_plan = planner().plan(tmp_path)
    action = action_plan.actions[0]
    bad_action = action.__class__(action.source, outside, action.category, action.reason)
    bad_plan = action_plan.__class__(tmp_path, (bad_action,), ())

    with pytest.raises(PermissionError):
        OrganizationExecutor().execute(bad_plan)

    assert source.exists()
    if outside.exists():
        outside.unlink()


def test_planner_does_not_process_directories(tmp_path):
    (tmp_path / "Pictures").mkdir()
    plan = planner().plan(tmp_path)
    assert not plan.actions
    assert not plan.skipped
