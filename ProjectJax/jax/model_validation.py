from __future__ import annotations


class ModelResponseError(ValueError):
    pass


def validate_chat_response(response: object) -> dict:
    if not isinstance(response, dict):
        raise ModelResponseError("Model response must be an object")
    message = response.get("message")
    if not isinstance(message, dict):
        raise ModelResponseError("Model response is missing a message object")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ModelResponseError("Model message content must be text")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ModelResponseError("tool_calls must be a list")
    for call in calls:
        if not isinstance(call, dict):
            raise ModelResponseError("Each tool call must be an object")
        function = call.get("function")
        if not isinstance(function, dict):
            raise ModelResponseError("Tool call is missing a function object")
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ModelResponseError("Tool call is missing a tool name")
        if not isinstance(arguments, dict):
            raise ModelResponseError(f"Arguments for {name} must be an object")
    return message
