from .executor import ExecutionRequest, ToolExecutor
from .result import ToolResult
from .model_validation import ModelResponseError, validate_chat_response
from .tasks import TaskManager
from .events import EventBus
from .session import SessionStore
import json

class Jax:
    def __init__(self, model, registry, policy, logger, max_turns=8, thinking_enabled=True, task_manager=None, event_bus=None, session_store=None, session_id=None):
        self.model = model
        self.registry = registry
        self.policy = policy
        self.logger = logger
        self.max_turns = max_turns
        self.tasks = task_manager or TaskManager()
        self.events = event_bus or EventBus()
        self.session_store = session_store
        self.session_id = session_id or (session_store.create() if session_store else None)
        self.thinking_enabled = thinking_enabled

    def run(self, user_text: str, approve_mutations=True) -> str:
        command = user_text.strip().lower()
        if command in {"stop thinking", "thinking off"}:
            self.thinking_enabled = False
            return "Thinking disabled."
        if command in {"start thinking", "thinking on"}:
            self.thinking_enabled = True
            return "Thinking enabled."
        if self.session_store is not None:
            self.session_store.append(self.session_id, "user", user_text)
        task = self.tasks.create(user_text)
        self.tasks.start(task)
        self.events.publish("task.started", task_id=task.id, request=user_text)
        try:
            result = self._run(task, approve_mutations)
            self.tasks.complete(task, result)
            if self.session_store is not None:
                self.session_store.append(self.session_id, "assistant", result)
            self.events.publish("task.completed", task_id=task.id, result=result)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.tasks.fail(task, error)
            self.events.publish("task.failed", task_id=task.id, error=error)
            raise

    def _run(self, task, approve_mutations=True) -> str:
        user_text = task.request
        authorized_root = getattr(self.policy, "authorized_root", None)
        root_instruction = (
            f"The authorized filesystem root is {authorized_root}. "
            if authorized_root is not None
            else "The filesystem policy did not expose an authorized root to the model. "
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are JAX, an autonomous Linux computer agent. {root_instruction}"
                    "The authorized root is your writable workspace. Work freely inside it. "
                    "Use the available tools to inspect, create, modify, execute, test, and "
                    "repair things as needed to accomplish the user's goal. Choose your own "
                    "sequence of actions; do not wait for approval between individual actions. "
                    "Observe every tool result and adapt when an action fails. Do not claim an "
                    "action occurred unless its result confirms it. Keep filesystem paths inside "
                    "the authorized root. Relative paths are resolved from that root; a relative path such as 'unsorted' means a child directory named unsorted. Do not turn a relative path into an absolute path beginning at '/' unless the user explicitly supplied that exact absolute path. "
                    "The run_command tool executes commands inside the workspace sandbox. "
                    "Prefer it when normal Linux tooling, scripts, shells, package managers, or "
                    "other programs are the best way to accomplish the task. Do not merely explain "
                    "how to perform an action when you can perform it with the available tools."
                ),
            },
            {"role": "user", "content": user_text},
        ]

        executor = ToolExecutor(self.registry, self.policy, self.logger, self.model, event_bus=self.events)
        for _ in range(self.max_turns):
            response = self.model.chat(messages, self.registry.specs(), think=self.thinking_enabled)
            try:
                assistant = validate_chat_response(response)
            except ModelResponseError as exc:
                return f"JAX received an invalid model response: {exc}"
            messages.append(assistant)

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                return assistant.get("content", "").strip()

            completed_in_response = {}
            for call in tool_calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = function.get("name")
                args = function.get("arguments") or {}
                if not isinstance(name, str) or not name:
                    result = ToolResult(False, error="Malformed tool call: missing tool name")
                    self.logger.record("<malformed>", args if isinstance(args, dict) else {}, result)
                else:
                    tool = self.registry.get(name)
                    request_key = None
                    if tool is not None and tool.mutates and isinstance(args, dict):
                        request_key = (name, json.dumps(args, sort_keys=True, default=str))
                    if request_key is not None and request_key in completed_in_response:
                        previous = completed_in_response[request_key]
                        result = ToolResult(True, {
                            "already_completed_in_response": True,
                            "original_result": previous.data,
                            "instruction": "This identical mutation was already completed in this model response. Continue with the remaining work.",
                        })
                    else:
                        result = executor.execute(
                            ExecutionRequest(name=name, arguments=args),
                            approve_mutations=approve_mutations,
                        )
                        if request_key is not None and result.ok:
                            completed_in_response[request_key] = result

                messages.append({
                    "role": "tool",
                    "tool_name": name or "<malformed>",
                    "content": self.model.encode_tool_result(result),
                })

        return "I stopped because the tool-call limit was reached."
