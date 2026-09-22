from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from .result import ToolResult
from .tools import FilesystemTool


class SearchFiles(FilesystemTool):
    name = "search_files"
    description = "Find files or directories by name pattern inside the authorized filesystem root. Relative search paths are resolved from the authorized root."
    mutates = False

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["pattern"],
        }

    def execute(self, pattern, path=".", limit=100):
        if not pattern or len(pattern) > 200:
            return ToolResult(False, error="Pattern must contain 1-200 characters")
        if limit < 1 or limit > 1000:
            return ToolResult(False, error="limit must be between 1 and 1000")
        root = self.safe(path)
        if not root.is_dir():
            return ToolResult(False, error=f"Not a directory: {root}")
        matches = []
        for item in root.rglob(pattern):
            if item.is_symlink():
                continue
            matches.append({"path": str(item), "type": "directory" if item.is_dir() else "file"})
            if len(matches) >= limit:
                break
        return ToolResult(True, {"matches": matches, "truncated": len(matches) >= limit})


class ReadTextFile(FilesystemTool):
    name = "read_text_file"
    description = "Read a bounded amount of UTF-8 text from a file inside the authorized root. Relative paths are resolved from the authorized root."
    mutates = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["path"]}

    def execute(self, path, max_bytes=65536):
        if max_bytes < 1 or max_bytes > 1024 * 1024:
            return ToolResult(False, error="max_bytes must be between 1 and 1048576")
        target = self.safe(path)
        if not target.is_file():
            return ToolResult(False, error=f"Not a file: {target}")
        try:
            raw = target.read_bytes()[:max_bytes]
            return ToolResult(True, {"path": str(target), "text": raw.decode("utf-8"), "truncated": target.stat().st_size > max_bytes})
        except UnicodeDecodeError:
            return ToolResult(False, error=f"File is not valid UTF-8 text: {target}")
        except OSError as exc:
            return ToolResult(False, error=f"Unable to read {target}: {exc}")


class WriteTextFile(FilesystemTool):
    name = "write_text_file"
    description = "Create a UTF-8 text file inside the authorized root without overwriting an existing file. Relative paths are resolved from the authorized root."
    mutates = True
    required_capabilities = {"filesystem.write"}

    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}, "required": ["path", "text"]}

    def execute(self, path, text):
        target = self.safe(path)
        if target.exists():
            return ToolResult(False, error=f"Destination already exists: {target}")
        if len(text.encode("utf-8")) > 1024 * 1024:
            return ToolResult(False, error="Text exceeds 1 MiB limit")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".jax-tmp")
            if temp.exists():
                return ToolResult(False, error=f"Temporary destination already exists: {temp}")
            temp.write_text(text, encoding="utf-8")
            temp.replace(target)
            return ToolResult(True, {"path": str(target), "bytes": target.stat().st_size})
        except OSError as exc:
            return ToolResult(False, error=f"Unable to write {target}: {exc}")


class SystemInfo:
    name = "system_info"
    description = "Report basic Linux/Python/system information."
    mutates = False
    required_capabilities = set()

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    def execute(self):
        return ToolResult(True, {
            "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "python": platform.python_version(), "cpu_count": os.cpu_count(),
        })


class DiskUsage(FilesystemTool):
    name = "disk_usage"
    description = "Report disk capacity and usage for a path inside the authorized root."
    mutates = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def execute(self, path="."):
        target = self.safe(path)
        if not target.exists():
            return ToolResult(False, error=f"Path does not exist: {target}")
        total, used, free = shutil.disk_usage(target)
        return ToolResult(True, {"path": str(target), "total": total, "used": used, "free": free})
