import json
from pathlib import Path

from jax.environment import discover_environment
from jax.events import EventBus
from jax.state import diff, snapshot


def test_snapshot_and_diff_detect_changes(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    before = snapshot(tmp_path)
    (tmp_path / "a.txt").write_text("two")
    (tmp_path / "b.txt").write_text("new")
    after = snapshot(tmp_path)
    result = diff(before, after)
    assert result["added"] == ["b.txt"]
    assert result["modified"] == ["a.txt"]


def test_snapshot_does_not_follow_symlink(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link"
    link.symlink_to(outside)
    state = snapshot(tmp_path)
    assert state.by_path()["link"].kind == "symlink"


def test_environment_discovery(tmp_path):
    (tmp_path / "Downloads").mkdir()
    env = discover_environment(tmp_path)
    assert env.home == tmp_path.resolve()
    assert env.downloads == (tmp_path / "Downloads").resolve()
    assert env.documents is None


def test_event_bus_specific_and_wildcard():
    seen = []
    bus = EventBus()
    bus.subscribe("task.created", lambda event: seen.append(("specific", event.data["id"])))
    bus.subscribe("*", lambda event: seen.append(("all", event.name)))
    event = bus.publish("task.created", id="123")
    assert event.name == "task.created"
    assert seen == [("specific", "123"), ("all", "task.created")]
