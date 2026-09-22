from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class ProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class ProcessPolicy:
    """Explicit boundary for commands JAX is allowed to run."""

    allowed_commands: frozenset[str]
    max_timeout: float = 30.0
    max_output: int = 64 * 1024

    def validate(self, argv: Sequence[str], timeout: float, cwd: Path | None) -> None:
        if not argv or not argv[0]:
            raise ProcessError("A command is required")
        command = Path(argv[0]).name
        if command not in self.allowed_commands:
            raise ProcessError(f"Command is not allowed: {command}")
        if timeout <= 0 or timeout > self.max_timeout:
            raise ProcessError(f"Timeout must be between 0 and {self.max_timeout} seconds")
        if cwd is not None and not cwd.is_dir():
            raise ProcessError(f"Working directory does not exist: {cwd}")


class ProcessRunner:
    def __init__(self, policy: ProcessPolicy):
        self.policy = policy

    def run(self, argv: Sequence[str], *, timeout: float = 10.0, cwd: Path | None = None,
            env: Mapping[str, str] | None = None) -> ProcessResult:
        argv = tuple(str(item) for item in argv)
        self.policy.validate(argv, timeout, cwd)
        safe_env = None
        if env is not None:
            safe_env = {str(k): str(v) for k, v in env.items()}
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd) if cwd else None,
                env=safe_env,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return ProcessResult(
                argv=argv,
                returncode=completed.returncode,
                stdout=completed.stdout[: self.policy.max_output],
                stderr=completed.stderr[: self.policy.max_output],
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ProcessResult(argv, -1, stdout[: self.policy.max_output], stderr[: self.policy.max_output], True)

    @staticmethod
    def parse_command(command: str) -> tuple[str, ...]:
        if not isinstance(command, str) or not command.strip():
            raise ProcessError("Command text is required")
        try:
            parts = tuple(shlex.split(command))
        except ValueError as exc:
            raise ProcessError(f"Invalid command syntax: {exc}") from exc
        if any("\x00" in part for part in parts):
            raise ProcessError("NUL bytes are not allowed")
        return parts


def linux_processes() -> list[dict[str, int | str]]:
    """Read basic process metadata without external commands."""
    proc = Path("/proc")
    if not proc.is_dir():
        raise ProcessError("/proc is unavailable")
    result = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(errors="replace")
            name = next((line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("Name:")), "")
            state = next((line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("State:")), "")
            result.append({"pid": int(entry.name), "name": name, "state": state})
        except (OSError, ValueError):
            continue
    return sorted(result, key=lambda item: int(item["pid"]))
