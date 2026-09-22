from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .recovery import RecoveryManager
from .verification import file_digest, verify_move


@dataclass(frozen=True)
class FileClassification:
    path: Path
    category: str | None
    confidence: str
    reason: str


@dataclass(frozen=True)
class OrganizationAction:
    source: Path
    destination: Path
    category: str
    reason: str


@dataclass(frozen=True)
class OrganizationPlan:
    root: Path
    actions: tuple[OrganizationAction, ...]
    skipped: tuple[FileClassification, ...]


class FileClassifier:
    """Deterministic baseline classifier. Rules are injected, not hard-coded into the organizer."""

    def __init__(self, rules: dict[str, set[str]]):
        self.rules = {category: {ext.lower() for ext in extensions} for category, extensions in rules.items()}

    def classify(self, path: Path) -> FileClassification:
        suffix = path.suffix.lower()
        matches = [category for category, extensions in self.rules.items() if suffix in extensions]
        if len(matches) == 1:
            category = matches[0]
            return FileClassification(path, category, "high", f"extension {suffix or '[none]'} matches {category}")
        if len(matches) > 1:
            return FileClassification(path, None, "ambiguous", "extension matches multiple categories")
        return FileClassification(path, None, "unknown", f"no rule for extension {suffix or '[none]'}")


class OrganizationPlanner:
    """Builds a non-mutating plan. It never moves files itself."""

    def __init__(self, classifier: FileClassifier):
        self.classifier = classifier

    def plan(self, root: Path) -> OrganizationPlan:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)

        actions: list[OrganizationAction] = []
        skipped: list[FileClassification] = []
        for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file():
                continue
            classification = self.classifier.classify(path)
            if classification.category is None:
                skipped.append(classification)
                continue
            destination = root / classification.category / path.name
            if destination.exists():
                skipped.append(FileClassification(
                    path, classification.category, "blocked",
                    f"destination already exists: {destination}",
                ))
                continue
            actions.append(OrganizationAction(
                path, destination, classification.category, classification.reason,
            ))
        return OrganizationPlan(root, tuple(actions), tuple(skipped))


class OrganizationExecutor:
    """Executes an already-approved plan and verifies every move."""

    def execute(self, plan: OrganizationPlan, recovery: RecoveryManager | None = None, dry_run: bool = False) -> tuple[OrganizationAction, ...]:
        if dry_run:
            return plan.actions

        completed: list[OrganizationAction] = []
        for action in plan.actions:
            source = action.source.resolve()
            destination = action.destination.resolve()
            if not _inside(plan.root, source) or not _inside(plan.root, destination):
                raise PermissionError("Organization action is outside the authorized root")
            if not source.is_file():
                raise FileNotFoundError(source)
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = file_digest(source)
            source.rename(destination)
            verify_move(source, destination, digest)
            if recovery is not None:
                recovery.record_move(source, destination)
            completed.append(action)
        return tuple(completed)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def default_rules() -> dict[str, set[str]]:
    return {
        "Documents": {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"},
        "Pictures": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"},
        "Projects": {".zip", ".tar", ".gz", ".tgz"},
    }
