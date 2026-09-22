from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class FileState:
    relative_path: str
    kind: str
    size: int | None
    mtime_ns: int | None
    sha256: str | None


@dataclass(frozen=True)
class FilesystemSnapshot:
    root: str
    files: tuple[FileState, ...]

    def by_path(self) -> dict[str, FileState]:
        return {item.relative_path: item for item in self.files}

    def to_dict(self) -> dict:
        return {"root": self.root, "files": [asdict(item) for item in self.files]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root: Path, *, hash_files: bool = True) -> FilesystemSnapshot:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    states: list[FileState] = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        # Never follow symlinked entries when describing filesystem state.
        if path.is_symlink():
            states.append(FileState(str(path.relative_to(root)), "symlink", None, None, None))
            continue
        relative = str(path.relative_to(root))
        if path.is_dir():
            states.append(FileState(relative, "directory", None, path.stat().st_mtime_ns, None))
        elif path.is_file():
            stat = path.stat()
            states.append(FileState(relative, "file", stat.st_size, stat.st_mtime_ns, _digest(path) if hash_files else None))
    return FilesystemSnapshot(str(root), tuple(states))


def diff(before: FilesystemSnapshot, after: FilesystemSnapshot) -> dict[str, list[str]]:
    if Path(before.root).resolve() != Path(after.root).resolve():
        raise ValueError("Snapshots must use the same root")
    old, new = before.by_path(), after.by_path()
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified = sorted(path for path in set(old) & set(new) if old[path] != new[path])
    return {"added": added, "removed": removed, "modified": modified}
