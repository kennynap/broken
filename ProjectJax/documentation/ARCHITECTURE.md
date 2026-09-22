# Jax Architecture Reference

Jax separates model decisions from deterministic execution authority.

## Main flow

```text
User request
    ↓
Jax task lifecycle
    ↓
Model request + tool specifications
    ↓
Model tool call
    ↓
ToolExecutor
    ↓
Argument validation
    ↓
Policy authorization
    ↓
Tool execution
    ↓
Result / event / log
    ↓
Model continues or returns
```

## Important components

| Component | Responsibility |
|---|---|
| `core.Jax` | Task execution loop, model/tool interaction, thinking state, session/task integration. |
| `model.OllamaModel` | Ollama `/api/chat` boundary and model metrics. |
| `tools` | Base tool types, registry, filesystem tools, and path handling. |
| `executor.ToolExecutor` | Single execution boundary: schema validation, policy check, mutation approval, execution, logging/events. |
| `policy.Policy` | Central capability and authorized-root authorization. |
| `sandbox.WorkspaceSandbox` | Bubblewrap process sandbox for `run_command`. |
| `planning.ToolPlan` | Preflight validation of a complete tool-call batch. |
| `verification` | SHA-256 file verification for filesystem moves. |
| `recovery` / `persistent_recovery` | Reversible move recovery support. |
| `tasks` / `task_store` | In-memory and persistent task lifecycle state. |
| `session` | Bounded persistent conversation history. |
| `events` | Structured lifecycle events. |
| `logging` | Persistent operation records. |
| `organizer` / `transactions` | Deterministic file classification, organization planning, verified execution, and transactional recovery. |

## Security boundary

The model is not the security boundary. Tool execution passes through deterministic checks. Filesystem paths are constrained to the authorized root. Process execution uses bubblewrap and refuses unsandboxed execution and root execution.

## Configuration

Relevant environment variables are defined in `jax/config.py`, including `JAX_OLLAMA_URL`, `JAX_MODEL`, `JAX_MAX_TURNS`, `JAX_THINKING`, `JAX_DEBUG`, `JAX_ALLOWED_ROOT`, and `JAX_CONFIRM_MUTATIONS`.

Inspect `Config` before relying on defaults or changing configuration behavior.
