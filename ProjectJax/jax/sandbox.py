from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class WorkspaceSandbox:
    """Run arbitrary shell commands inside a dedicated workspace sandbox.

    The sandbox uses bubblewrap when available. The workspace is mounted read/write
    at /workspace; the host filesystem is otherwise exposed read-only only where
    needed to execute normal Linux programs. The caller should run JAX as a
    dedicated non-privileged user with no sudo access.
    """

    def __init__(self, workspace: Path, *, max_timeout: float = 300.0, max_output: int = 256 * 1024):
        self.workspace = workspace.expanduser().resolve()
        self.max_timeout = max_timeout
        self.max_output = max_output
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.bwrap = shutil.which("bwrap") or shutil.which("bubblewrap")

    def require_available(self) -> None:
        if not self.bwrap:
            raise SandboxError(
                "bubblewrap (bwrap) is required for workspace process execution; "
                "refusing to run unsandboxed"
            )
        if os.geteuid() == 0:
            raise SandboxError("JAX must not run as root")

    def run(self, command: str, *, timeout: float = 60.0, env: Mapping[str, str] | None = None) -> SandboxResult:
        self.require_available()
        if not isinstance(command, str) or not command.strip():
            raise SandboxError("command must be a non-empty string")
        if timeout <= 0 or timeout > self.max_timeout:
            raise SandboxError(f"timeout must be between 0 and {self.max_timeout} seconds")

        self._prepare_workspace()
        child_env = {
            "HOME": "/workspace/home",
            "PATH": "/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", os.environ.get("LANG", "C.UTF-8")),
            "PWD": "/workspace",
        }
        if env:
            child_env.update({str(k): str(v) for k, v in env.items()})

        argv = [
            self.bwrap,
            "--die-with-parent",
            "--new-session",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/sbin", "/sbin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind", "/sys", "/sys",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(self.workspace), "/workspace",
            "--chdir", "/workspace",
            "--setenv", "HOME", child_env["HOME"],
            "--setenv", "PATH", child_env["PATH"],
            "--setenv", "LANG", child_env["LANG"],
            "--setenv", "LC_ALL", child_env["LC_ALL"],
            "--setenv", "PWD", "/workspace",
            "/bin/bash", "-lc", command,
        ]
        try:
            completed = subprocess.run(
                argv,
                cwd=str(self.workspace),
                env=child_env,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return SandboxResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout[: self.max_output],
                stderr=completed.stderr[: self.max_output],
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return SandboxResult(command, -1, stdout[: self.max_output], stderr[: self.max_output], True)
        except OSError as exc:
            raise SandboxError(f"unable to start sandbox: {exc}") from exc

    def _prepare_workspace(self) -> None:
        (self.workspace / "home").mkdir(exist_ok=True)
        (self.workspace / "bin").mkdir(exist_ok=True)
