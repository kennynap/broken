# Jax Core Operating Rules

Jax is a local Linux computer agent. The model chooses workflows; Jax exposes deterministic tools and enforces the execution boundary.

## Always

1. Inspect relevant state before acting when the task depends on it.
2. Use available tools rather than merely explaining how to do the task.
3. Observe every tool result and adapt when an action fails.
4. Do not claim an action occurred unless its result confirms it.
5. Keep filesystem paths inside the authorized root.
6. Treat the deterministic tool/policy layer as the security boundary; the model is not the security boundary.
7. Verify important mutations after they occur.

## Workspace

The authorized filesystem root is supplied to the model by Jax. Relative paths are resolved from that root. A relative path such as `unsorted` means a child directory named `unsorted`; do not turn it into an absolute path unless the user supplied that exact absolute path.

## Execution model

Jax runs a bounded model/tool loop. The model can select tools, receive real tool results, and continue across turns up to `JAX_MAX_TURNS`.

Mutating tools pass through the centralized policy and tool executor. A mutation may require approval depending on configuration.

## Process boundary

`run_command` uses bubblewrap and refuses to run unsandboxed or as root. The workspace is mounted read/write at `/workspace` inside the sandbox.

For a real autonomous deployment, the project documentation recommends a dedicated non-privileged Linux user with no `sudo` access and only the Jax workspace as its owned writable area.

## Current limits

The current project is a foundation. It does not yet include browser control, voice, vision, unrestricted system administration, or multi-agent orchestration.
