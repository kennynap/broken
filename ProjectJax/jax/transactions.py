from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .organizer import OrganizationAction, OrganizationPlan
from .recovery import RecoveryManager
from .verification import file_digest, verify_move


@dataclass(frozen=True)
class TransactionResult:
    dry_run: bool
    completed: tuple[OrganizationAction, ...]
    rolled_back: bool = False


class OrganizationTransaction:
    """Execute an organization plan as a verified batch with optional rollback."""

    def __init__(self, recovery: RecoveryManager):
        self.recovery = recovery

    def run(
        self,
        plan: OrganizationPlan,
        *,
        dry_run: bool = False,
        atomic: bool = True,
        fail_after: int | None = None,
    ) -> TransactionResult:
        if dry_run:
            self._validate_plan(plan)
            return TransactionResult(True, plan.actions)

        self._validate_plan(plan)
        completed: list[OrganizationAction] = []
        try:
            for index, action in enumerate(plan.actions, start=1):
                self._move(action, plan.root)
                completed.append(action)
                if fail_after is not None and index >= fail_after:
                    raise RuntimeError("Injected transaction failure for testing")
        except Exception:
            if atomic:
                self._rollback(len(completed))
                return TransactionResult(False, tuple(completed), True)
            raise

        return TransactionResult(False, tuple(completed), False)

    def _validate_plan(self, plan: OrganizationPlan) -> None:
        root = plan.root.resolve()
        seen_destinations: set[Path] = set()
        for action in plan.actions:
            source = action.source.resolve()
            destination = action.destination.resolve()
            self._inside(root, source)
            self._inside(root, destination)
            if source == destination:
                raise ValueError(f"Source and destination are identical: {source}")
            if destination in seen_destinations:
                raise ValueError(f"Duplicate destination in plan: {destination}")
            seen_destinations.add(destination)
            if not source.is_file():
                raise FileNotFoundError(source)
            if destination.exists():
                raise FileExistsError(destination)

    def _move(self, action: OrganizationAction, root: Path) -> None:
        source = action.source.resolve()
        destination = action.destination.resolve()
        self._inside(root.resolve(), source)
        self._inside(root.resolve(), destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = file_digest(source)
        source.rename(destination)
        verify_move(source, destination, digest)
        self.recovery.record_move(source, destination)

    def _rollback(self, count: int) -> None:
        for _ in range(count):
            self.recovery.undo_last()

    @staticmethod
    def _inside(root: Path, candidate: Path) -> None:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"Path outside authorized root: {candidate}") from exc
