from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .result import ToolResult
from .schema import ToolArgumentError, validate_tool_arguments


@dataclass(frozen=True)
class ExecutionRequest:
    name: str
    arguments: dict[str, Any]


class ToolExecutor:
    """Single execution boundary for registered JAX tools."""

    def __init__(self, registry, policy, logger, model=None, event_bus=None):
        self.registry = registry
        self.policy = policy
        self.logger = logger
        self.model = model
        self.events = event_bus

    def execute(self, request: ExecutionRequest, approve_mutations: bool = True) -> ToolResult:
        if self.events:
            self.events.publish("tool.requested", tool=request.name, arguments=request.arguments)
        tool = self.registry.get(request.name)
        if tool is None:
            return self._finish(request, ToolResult(False, error=f"Unknown tool: {request.name}"))

        if not isinstance(request.arguments, dict):
            return self._finish(request, ToolResult(False, error="Tool arguments must be an object"))

        try:
            validate_tool_arguments(tool, request.arguments)
        except ToolArgumentError as exc:
            return self._finish(request, ToolResult(False, error=str(exc)))

        decision = self.policy.check(tool, request.arguments)
        if not decision.allowed:
            return self._finish(request, ToolResult(False, error=f"Permission denied for tool: {request.name}: {decision.reason}"))

        if tool.mutates:
            if not approve_mutations:
                return self._finish(request, ToolResult(False, error=f"Approval required for tool: {request.name}"))
            if not self.policy.confirm(tool, request.arguments):
                return self._finish(request, ToolResult(False, error="User did not approve the operation."))

        try:
            result = tool.execute(**request.arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult(False, error=f"Tool {request.name} returned an invalid result")
        except TypeError as exc:
            result = ToolResult(False, error=f"Invalid arguments for tool {request.name}: {exc}")
        except Exception as exc:
            result = ToolResult(False, error=f"{type(exc).__name__}: {exc}")

        return self._finish(request, result)

    def _finish(self, request: ExecutionRequest, result: ToolResult) -> ToolResult:
        self.logger.record(request.name, request.arguments if isinstance(request.arguments, dict) else {}, result)
        if self.events:
            self.events.publish("tool.completed", tool=request.name, ok=result.ok, error=result.error)
        return result
