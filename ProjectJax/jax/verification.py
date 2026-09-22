from __future__ import annotations

import hashlib
from pathlib import Path


class VerificationError(RuntimeError):
    pass


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_move(source: Path, destination: Path, expected_digest: str | None = None) -> None:
    if source.exists():
        raise VerificationError(f"Move verification failed: source still exists: {source}")
    if not destination.is_file():
        raise VerificationError(f"Move verification failed: destination is missing: {destination}")
    if expected_digest is not None and file_digest(destination) != expected_digest:
        raise VerificationError(f"Move verification failed: destination content differs: {destination}")
