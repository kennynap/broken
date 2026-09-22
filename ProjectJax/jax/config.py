from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass(frozen=True)
class Config:
    ollama_url: str = field(default_factory=lambda: os.getenv("JAX_OLLAMA_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("JAX_MODEL", "qwen3:8b"))
    max_turns: int = field(default_factory=lambda: int(os.getenv("JAX_MAX_TURNS", "8")))
    thinking_enabled: bool = field(default_factory=lambda: os.getenv("JAX_THINKING", "true").strip().lower() in {"1", "true", "yes", "on"})
    debug: bool = field(default_factory=lambda: os.getenv("JAX_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"})
    allowed_root: Path = field(default_factory=lambda: Path(os.getenv("JAX_ALLOWED_ROOT", str(Path.home() / "JAX_WORKSPACE"))).expanduser().resolve())
    confirm_mutations: bool = field(default_factory=lambda: os.getenv("JAX_CONFIRM_MUTATIONS", "false").strip().lower() in {"1", "true", "yes", "on"})

    def validate(self) -> None:
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError("JAX_OLLAMA_URL must use http:// or https://")
        if not self.model.strip():
            raise ValueError("JAX_MODEL cannot be empty")
        if self.max_turns < 1 or self.max_turns > 100:
            raise ValueError("JAX_MAX_TURNS must be between 1 and 100")
        if self.allowed_root == Path("/"):
            raise ValueError("JAX_ALLOWED_ROOT cannot be the filesystem root")
