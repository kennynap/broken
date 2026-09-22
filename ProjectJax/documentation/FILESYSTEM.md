# Jax Filesystem Rules

## Boundary

Jax tools are authorized against one configured filesystem root. The default configured root is `~/JAX_WORKSPACE`. It can be changed with `JAX_ALLOWED_ROOT`.

The policy layer checks path arguments centrally, and filesystem tools perform their own path resolution/checking. Paths outside the configured root are rejected.

## Paths

Relative paths are interpreted from the authorized root.

Examples:

```text
unsorted/file.txt
./project/file.py
```

Do not convert a relative path into `/...` unless the user explicitly supplied that absolute path.

## Mutation rules

- Existing destination files are not overwritten by the built-in create/move/rename operations.
- `write_text_file` creates a new file and refuses an existing destination.
- `move_file` refuses an existing destination.
- Filesystem mutation requires the `filesystem.write` capability and passes through the centralized policy/executor.

## Reading/searching

`read_text_file` reads UTF-8 text up to its configured maximum. `search_files` searches within the authorized root and skips symlinks.

## Verification

File moves can be verified with SHA-256 content digests. A verified move requires the source to be absent, the destination to exist, and—when an expected digest is supplied—the destination content to match it.

## Do not assume

A path existing in documentation, a previous result, or a model response does not prove it currently exists. Inspect the actual filesystem when current state matters.
