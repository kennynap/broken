# Jax Self-Development

## Goal

Jax is intended to eventually work on its own code while a human verifies consequential changes.

The target workflow is:

```text
request
  ↓
inspect implementation
  ↓
inspect relevant tests/documentation
  ↓
plan the smallest change
  ↓
modify code
  ↓
run tests
  ↓
diagnose failures
  ↓
repair if justified
  ↓
verify behavior
  ↓
present the result for human verification
```

## Current capability

The current agent can use filesystem and process tools in its authorized workspace. `run_command` can build and test software inside its workspace sandbox when the required programs are available there.

This does **not** mean the current system has autonomous self-development as a completed feature. The workflow above is the development target.

## Rules for self-modification

- Never assume the repository structure.
- Inspect actual files before editing.
- Preserve existing interfaces unless the task requires a change.
- Prefer the smallest change that satisfies the task.
- Run tests after meaningful changes.
- Do not claim tests passed unless they actually passed.
- Keep security enforcement in deterministic Jax code rather than relying on model instructions.
- Do not weaken a security control merely to make a development task easier.
- Keep documentation synchronized when a code change changes knowledge Jax needs to operate correctly.

## Human verification

Model reasoning is not authorization. A future self-development workflow should make proposed changes inspectable and leave the human as the final verifier for consequential modifications.
