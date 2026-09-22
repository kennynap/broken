from __future__ import annotations

from .process import linux_processes
from .result import ToolResult
from .sandbox import SandboxError, WorkspaceSandbox


class ListProcesses:
    name = "list_processes"
    description = "List basic information about currently running Linux processes."
    mutates = False
    required_capabilities = {"process.read"}

    @property
    def parameters(self):
        return {"type": "object", "properties": {"limit": {"type": "integer"}}}

    def execute(self, limit=100):
        if limit < 1 or limit > 1000:
            return ToolResult(False, error="limit must be between 1 and 1000")
        try:
            processes = linux_processes()[:limit]
        except Exception as exc:
            return ToolResult(False, error=f"Unable to inspect processes: {exc}")
        return ToolResult(True, {"processes": processes, "truncated": len(processes) >= limit})


class RunCommand:
    name = "run_command"
    description = (
        "Run an arbitrary bash command inside the JAX workspace. The workspace is "
        "the agent's writable environment. Commands may create, modify, delete, build, "
        "test, install user-level packages, and execute programs there. The sandbox returns "
        "the real exit code, stdout, stderr, and timeout state."
    )
    mutates = True
    required_capabilities = {"process.execute"}

    def __init__(self, workspace):
        self.sandbox = WorkspaceSandbox(workspace)

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
            },
            "required": ["command"],
        }

    def execute(self, command, timeout=60.0):
        try:
            result = self.sandbox.run(command, timeout=float(timeout))
        except SandboxError as exc:
            return ToolResult(False, error=str(exc))
        return ToolResult(True, {
            "command": result.command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        })
