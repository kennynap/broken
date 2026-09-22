from abc import ABC, abstractmethod
from pathlib import Path
from .result import ToolResult

class BaseTool(ABC):
    name = ""
    description = ""
    mutates = False
    required_capabilities = {"filesystem.read"}

    @property
    @abstractmethod
    def parameters(self):
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        ...

    def spec(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

class ToolRegistry:
    def __init__(self, tools):
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name):
        return self._tools.get(name)

    def specs(self):
        return [tool.spec() for tool in self._tools.values()]

class FilesystemTool(BaseTool):
    def __init__(self, root: Path):
        self.root = root.resolve()

    def safe(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"Path outside authorized root: {candidate}")
        return candidate

class ListDirectory(FilesystemTool):
    name = "list_directory"
    description = "List files and directories inside the authorized filesystem root or a child directory. Relative paths are resolved from the authorized root; prefer relative paths such as 'unsorted'."
    mutates = False
    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def execute(self, path):
        p = self.safe(path)
        if not p.is_dir():
            return ToolResult(False, error=f"Not a directory: {p}")
        items = []
        for x in sorted(p.iterdir(), key=lambda v: v.name.lower()):
            items.append({"name": x.name, "type": "directory" if x.is_dir() else "file", "size": x.stat().st_size if x.is_file() else None})
        return ToolResult(True, {"path": str(p), "items": items})

class InspectFile(FilesystemTool):
    name = "inspect_file"
    description = "Inspect basic metadata for a file inside the authorized filesystem root. Relative paths are resolved from the authorized root."
    mutates = False
    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def execute(self, path):
        p = self.safe(path)
        if not p.is_file():
            return ToolResult(False, error=f"Not a file: {p}")
        s = p.stat()
        return ToolResult(True, {
            "path": str(p),
            "name": p.name,
            "suffix": p.suffix.lower(),
            "size": s.st_size,
            "modified": s.st_mtime,
        })

class CreateDirectory(FilesystemTool):
    name = "create_directory"
    description = "Create a directory inside the authorized filesystem root. Relative paths are resolved from the authorized root. Does not delete or overwrite anything."
    mutates = True
    required_capabilities = {"filesystem.write"}
    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def execute(self, path):
        p = self.safe(path)
        try:
            p.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return ToolResult(False, error=f"Destination already exists: {p}")
        except OSError as exc:
            return ToolResult(False, error=f"Unable to create directory {p}: {exc}")
        return ToolResult(True, {"created": str(p)})

class MoveFile(FilesystemTool):
    name = "move_file"
    description = "Move a file inside the authorized filesystem root. Relative source and destination paths are resolved from the authorized root. Never overwrites an existing destination."
    mutates = True
    required_capabilities = {"filesystem.write"}
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["source", "destination"],
        }

    def execute(self, source, destination):
        src, dst = self.safe(source), self.safe(destination)
        if not src.is_file():
            return ToolResult(False, error=f"Source is not a file: {src}")
        if dst.exists():
            return ToolResult(False, error=f"Destination already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return ToolResult(True, {"source": str(src), "destination": str(dst)})

class RenameFile(FilesystemTool):
    name = "rename_file"
    description = "Rename a file inside its current directory. Relative paths are resolved from the authorized root. Never overwrites an existing file."
    mutates = True
    required_capabilities = {"filesystem.write"}
    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}, "new_name": {"type": "string"}}, "required": ["path", "new_name"]}

    def execute(self, path, new_name):
        src = self.safe(path)
        if not src.is_file():
            return ToolResult(False, error=f"Not a file: {src}")
        if Path(new_name).name != new_name:
            return ToolResult(False, error="new_name must be a filename, not a path")
        dst = src.with_name(new_name)
        if dst.exists():
            return ToolResult(False, error=f"Destination already exists: {dst}")
        src.rename(dst)
        return ToolResult(True, {"source": str(src), "destination": str(dst)})

class VerifyPath(FilesystemTool):
    name = "verify_path"
    description = "Verify whether a path exists and report whether it is a file or directory. Relative paths are resolved from the authorized root."
    mutates = False
    @property
    def parameters(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def execute(self, path):
        p = self.safe(path)
        return ToolResult(True, {"path": str(p), "exists": p.exists(), "type": "directory" if p.is_dir() else "file" if p.is_file() else None})
