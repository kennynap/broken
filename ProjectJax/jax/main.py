from pathlib import Path
from .config import Config
from .core import Jax
from .logging import OperationLogger
from .model import OllamaModel, format_metrics
from .policy import Policy
from .tools import ToolRegistry
from .builtins import filesystem_tools

def build_jax(config=None):
    config = config or Config()
    config.validate()
    config.allowed_root.mkdir(parents=True, exist_ok=True)

    # The model is the agent. JAX provides capabilities; Linux/sandboxing provides the boundary.
    tools = filesystem_tools(config.allowed_root)

    return Jax(
        model=OllamaModel(config.ollama_url, config.model),
        registry=ToolRegistry(tools),
        policy=Policy(config.allowed_root, confirm_mutations=config.confirm_mutations),
        logger=OperationLogger(Path.home() / ".local" / "share" / "jax" / "operations.jsonl"),
        max_turns=config.max_turns,
        thinking_enabled=config.thinking_enabled,
    )

def main():
    config = Config()
    config.validate()
    print("JAX — controlled filesystem assistant")
    print("Authorized root:", config.allowed_root)
    print("Type 'exit' to quit.")

    jax = build_jax(config)
    while True:
        try:
            request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if request.lower() in {"exit", "quit"}:
            break
        if not request:
            continue

        try:
            model = jax.model
            calls_before = model.request_count
            result = jax.run(request)
            print("JAX:", result)
            if model.last_metrics is not None:
                print(format_metrics(model.last_metrics, model.request_count - calls_before))
            if config.debug:
                print("[Debug tools: " + ", ".join(tool.name for tool in jax.registry._tools.values()) + "]")
        except Exception as exc:
            print(f"JAX error: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
