from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


class Policy:
    """Central authorization boundary for JAX tool execution.

    Tool implementations still validate their own arguments. This layer is a
    second, centralized check that prevents tools from operating outside the
    configured root or without the required capability.
    """

    DEFAULT_CAPABILITIES = frozenset({"filesystem.read", "filesystem.write", "process.read", "process.execute"})

    def __init__(
        self,
        authorized_root: Path,
        input_fn: Callable[[str], str] = input,
        capabilities: Iterable[str] | None = None,
        confirm_mutations: bool = True,
    ):
        self.authorized_root = authorized_root.expanduser().resolve()
        self.input_fn = input_fn
        self.capabilities = frozenset(self.DEFAULT_CAPABILITIES if capabilities is None else capabilities)
        self.confirm_mutations = confirm_mutations

    def check(self, tool, args: dict) -> PermissionDecision:
        required = frozenset(getattr(tool, "required_capabilities", set()))
        missing = required - self.capabilities
        if missing:
            return PermissionDecision(False, f"Missing capabilities: {', '.join(sorted(missing))}")

        for path in self._path_arguments(args):
            if not self._inside_root(path):
                return PermissionDecision(False, f"Path outside authorized root: {path}")

        return PermissionDecision(True, "allowed")

    def allowed(self, tool, args: dict) -> bool:
        return self.check(tool, args).allowed

    def confirm(self, tool, args):
        if not self.confirm_mutations:
            return True
        print(f"JAX requests: {tool.name} {args}")
        return self.input_fn("Approve? [y/N]: ").strip().lower() in {"y", "yes"}

    def _path_arguments(self, args: dict):
        for key, value in args.items():
            if key not in {"path", "source", "destination"}:
                continue
            if not isinstance(value, str):
                continue
            yield self._resolve(value)

    def _resolve(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.authorized_root / candidate
        return candidate.resolve()

    def _inside_root(self, candidate: Path) -> bool:
        if candidate == self.authorized_root:
            return True
        try:
            candidate.relative_to(self.authorized_root)
            return True
        except ValueError:
            return False
