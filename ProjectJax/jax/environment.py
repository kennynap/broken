from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentPaths:
    home: Path
    desktop: Path | None
    downloads: Path | None
    documents: Path | None
    pictures: Path | None
    videos: Path | None
    music: Path | None


def _optional(home: Path, name: str) -> Path | None:
    candidate = home / name
    return candidate if candidate.is_dir() else None


def discover_environment(home: Path | None = None) -> EnvironmentPaths:
    home = (home or Path.home()).expanduser().resolve()
    return EnvironmentPaths(
        home=home,
        desktop=_optional(home, "Desktop"),
        downloads=_optional(home, "Downloads"),
        documents=_optional(home, "Documents"),
        pictures=_optional(home, "Pictures"),
        videos=_optional(home, "Videos"),
        music=_optional(home, "Music"),
    )
