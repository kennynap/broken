# Jax Coding Workflow

Use this workflow when changing Jax or another codebase in the authorized workspace.

1. **Inspect** the relevant implementation and project structure.
2. **Inspect tests** that cover the behavior.
3. **Establish a baseline** before changing behavior when practical.
4. **Make the smallest necessary change.**
5. **Inspect the resulting code** and its interfaces/imports.
6. **Run relevant tests.**
7. **Verify behavior**, not just syntax.
8. **Review the final package** when packaging or publishing is involved.
9. Report what was actually verified and identify anything not verified.

## Code changes

Do not invent filenames, dependencies, commands, interfaces, or project structure. Inspect first.

Do not change unrelated architecture while fixing a focused problem.

When a dependency or command is required, establish it from the actual project metadata/environment before relying on it.

## Tests

The project uses `tests/` as its pytest test path. The repository currently declares no Python runtime dependencies in `pyproject.toml`; do not infer that a test runner or optional dependency is installed outside the inspected environment.

## Self-modification

Jax may eventually use this workflow to work on its own source. The intended sequence is still inspect → change → test → verify. Human verification remains the approval point for consequential changes; do not treat model confidence as authorization.
