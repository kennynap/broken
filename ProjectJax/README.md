# JAX

Initial controlled AI/computer foundation.

## Foundation status

The deterministic foundation is implemented and tested. It includes:

- JAX core and task lifecycle
- Ollama-compatible model boundary
- model-response validation
- tool registry and dedicated tool executor
- centralized permission policy
- authorized filesystem root and path/symlink protection
- filesystem inspection and controlled mutation tools
- deterministic file classification and organization planning
- collision protection
- post-operation content verification
- reversible filesystem moves with integrity checks
- persistent JSONL operation logging
- isolated automated and integration tests

## Safety boundary

By default JAX is authorized only to operate inside:

`$HOME/Downloads`

Change it with:

`JAX_ALLOWED_ROOT=/path/to/root`

Mutation tools require interactive approval. Existing destination files are never overwritten.

## Run

The project uses only the Python standard library at this stage.

```bash
python -m jax.main
```

The real Ollama integration requires an Ollama server and a configured model. That external integration is not claimed as verified by the local test suite.

## Tests

```bash
python -m pytest
```

The test suite uses temporary isolated directories and never uses real user files.

## Architecture boundary

OpenJarvis is being used as an architectural reference and potential source of reusable infrastructure, not as the identity of this project. Its current documentation describes `BaseAgent`, `ToolUsingAgent`, `ToolExecutor`, structured tools, and an Ollama engine; those are candidates for selective reuse when integration can be made stable and justified. JAX-specific policy, filesystem authority, verification, recovery, task state, and personal-environment behavior remain under JAX control.

## Deliberately not included yet

- browser control
- shell execution
- voice
- vision
- autonomous long-running workflows
- persistent conversational memory
- multi-agent orchestration
- unrestricted system administration
- real-world Ollama end-to-end verification

## Current foundation additions

The foundation also includes:

- filesystem snapshots and content-aware diffs
- local Linux environment path discovery
- persistent task state with restart recovery
- structured event bus for task/tool lifecycle events
- deterministic tool argument/schema validation
- configuration validation for unsafe or invalid settings

## Current verification

The complete local suite currently contains 56 passing tests. The implementation has also been byte-compiled and exercised through isolated filesystem/manual integration checks. Real Ollama behavior remains intentionally unverified until an Ollama server/model is available.

## Extended deterministic capabilities

The current foundation additionally includes:

- bounded filesystem search and UTF-8 text inspection
- safe text-file creation without overwrite
- basic system information and disk-usage inspection
- Linux process inventory through `/proc` without an external package
- an explicit subprocess boundary with command allowlisting, no shell invocation, timeouts, and output limits
- persistent bounded conversation sessions
- complete tool-plan preflight before mutation execution
- optional session persistence integrated into the JAX core

These capabilities use only the Python standard library and are covered by isolated tests.
