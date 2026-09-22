from __future__ import annotations

from typing import Any


class ToolArgumentError(ValueError):
    pass


def validate_tool_arguments(tool: Any, arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolArgumentError("Tool arguments must be an object")
    schema = getattr(tool, "parameters", None)
    # Legacy/internal tools may not expose a schema yet. Preserve their
    # execution contract while requiring schemas for tools that provide one.
    if schema is None:
        return
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ToolArgumentError(f"Tool {tool.name} has an invalid parameter schema")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    unknown = set(arguments) - set(properties)
    if unknown:
        raise ToolArgumentError(f"Unknown arguments for {tool.name}: {', '.join(sorted(unknown))}")
    missing = set(required) - set(arguments)
    if missing:
        raise ToolArgumentError(f"Missing arguments for {tool.name}: {', '.join(sorted(missing))}")
    for name, value in arguments.items():
        expected = properties[name].get("type") if isinstance(properties[name], dict) else None
        if expected == "string" and not isinstance(value, str):
            raise ToolArgumentError(f"Argument {name} must be a string")
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ToolArgumentError(f"Argument {name} must be an integer")
        if expected == "boolean" and not isinstance(value, bool):
            raise ToolArgumentError(f"Argument {name} must be a boolean")
