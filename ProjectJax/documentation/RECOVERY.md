# Jax Recovery and Verification

Recovery means responding to real tool results, not guessing why an operation failed.

## On failure

1. Read the tool error/result.
2. Determine what actually changed, if anything.
3. Inspect current state when necessary.
4. Correct the cause with the smallest appropriate action.
5. Retry only when the new action is justified.
6. Verify the resulting state.
7. Report the outcome accurately.

## Tool plans

`ToolPlan` can preflight a batch of calls before mutations execute. It validates tool names, arguments, and policy authorization for the complete batch. If preflight fails, the batch does not begin mutation execution.

## Filesystem moves

Jax has recovery support for recorded filesystem moves. Undo is refused when the original source exists, the recorded destination is missing, or the destination content has changed from the recorded SHA-256 digest.

Organization transactions can run as dry runs and can roll back completed moves when configured as atomic and a later operation fails.

## Tasks

Tasks have four states: `created`, `running`, `completed`, and `failed`. Persistent task state can survive process restarts; tasks left running by a previous process can be marked failed as interrupted.

## Verification principle

Never infer success from the absence of an exception alone. Use the returned result and, when the operation requires stronger assurance, inspect the resulting state or use the project's verification mechanism.
