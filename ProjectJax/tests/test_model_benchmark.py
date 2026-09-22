from pathlib import Path

from scripts.model_benchmark import EXPECTED_FILES, evaluate, reset_sandbox


class Result:
    def __init__(self, ok):
        self.ok = ok
        self.error = None


class Logger:
    records = []


def test_reset_sandbox_creates_known_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    root = tmp_path / "Downloads" / ".jax-model-benchmark-sandbox"
    reset_sandbox(root)
    assert sorted(p.name for p in (root / "unsorted").iterdir()) == sorted(EXPECTED_FILES)


def test_evaluate_knowledge_requires_expected_answer(tmp_path):
    assert evaluate("knowledge", "Paris is the capital of France.", tmp_path, Logger())
    assert not evaluate("knowledge", "London is the capital of France.", tmp_path, Logger())


def test_evaluate_organize_checks_actual_filesystem(tmp_path):
    root = tmp_path
    for name, category in EXPECTED_FILES.items():
        destination = root / category
        destination.mkdir(parents=True, exist_ok=True)
        (destination / name).write_text("fixture")
    assert evaluate("organize", "completed", root, Logger())
    (root / "unsorted").mkdir()
    (root / "unsorted" / "data.csv").write_text("still here")
    assert not evaluate("organize", "completed", root, Logger())
