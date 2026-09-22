from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .executor import ExecutionRequest, ToolExecutor
from .result import ToolResult


@dataclass(frozen=True)
class PlannedCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PlanCheck:
    allowed: bool
    errors: tuple[str, ...]


class ToolPlan:
    """Validate a complete set of tool calls before any mutation executes."""

    def __init__(self, calls: list[PlannedCall]):
        self.calls = tuple(calls)

    def preflight(self, executor: ToolExecutor) -> PlanCheck:
        errors = []
        for index, call in enumerate(self.calls):
            tool = executor.registry.get(call.name)
            if tool is None:
                errors.append(f"#{index + 1}: unknown tool {call.name}")
                continue
            try:
                from .schema import validate_tool_arguments
                validate_tool_arguments(tool, call.arguments)
            except Exception as exc:
                errors.append(f"#{index + 1}: {exc}")
                continue
            decision = executor.policy.check(tool, call.arguments)
            if not decision.allowed:
                errors.append(f"#{index + 1}: {decision.reason}")
        return PlanCheck(not errors, tuple(errors))

    def execute(self, executor: ToolExecutor, *, approve_mutations: bool = True) -> tuple[ToolResult, ...]:
        check = self.preflight(executor)
        if not check.allowed:
            return tuple(ToolResult(False, error=error) for error in check.errors)
        results = []
        for call in self.calls:
            result = executor.execute(ExecutionRequest(call.name, call.arguments), approve_mutations=approve_mutations)
            results.append(result)
            if not result.ok:
                break
        return tuple(results)
