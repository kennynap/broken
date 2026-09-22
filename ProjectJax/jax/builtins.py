from __future__ import annotations

from pathlib import Path
from .process import ProcessPolicy, ProcessRunner
from .system_tools import SearchFiles, ReadTextFile, WriteTextFile, SystemInfo, DiskUsage
from .process_tools import ListProcesses, RunCommand
from .tools import ListDirectory, InspectFile, CreateDirectory, MoveFile, RenameFile, VerifyPath


def filesystem_tools(root: Path):
    return [
        ListDirectory(root), InspectFile(root), CreateDirectory(root), MoveFile(root),
        RenameFile(root), VerifyPath(root), SearchFiles(root), ReadTextFile(root),
        WriteTextFile(root), DiskUsage(root), SystemInfo(), ListProcesses(), RunCommand(root),
    ]


def default_process_runner() -> ProcessRunner:
    return ProcessRunner(ProcessPolicy(frozenset({"pwd", "uname", "whoami", "true"})))
