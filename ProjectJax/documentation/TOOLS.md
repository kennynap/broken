# Jax Tools

Tool descriptions are the model-facing interface. Keep them short and precise.

| Tool | Purpose | Important boundary |
|---|---|---|
| `list_directory` | List files/directories. | Authorized root only. |
| `inspect_file` | Inspect basic file metadata. | Authorized root only. |
| `create_directory` | Create a directory. | No overwrite; authorized root. |
| `move_file` | Move a file. | No destination overwrite; authorized root. |
| `rename_file` | Rename a file in its current directory. | No overwrite; authorized root. |
| `verify_path` | Verify a path within the authorized filesystem boundary. | Authorized root only. |
| `search_files` | Find files/directories by name pattern. | Authorized root; does not follow symlinks. |
| `read_text_file` | Read bounded UTF-8 text. | Authorized root; maximum 1 MiB. |
| `write_text_file` | Create a UTF-8 text file. | Does not overwrite; maximum 1 MiB. |
| `disk_usage` | Report disk capacity/usage for a path. | Authorized root only. |
| `system_info` | Report basic Linux/Python/system information. | Read-only. |
| `list_processes` | List basic running Linux processes. | Read-only; limit 1–1000. |
| `run_command` | Run an arbitrary Bash command in the workspace sandbox. | Requires bubblewrap; refuses root/unsandboxed execution; timeout is bounded. |

## Tool workflow

- Use inspection/read tools to establish facts before dependent actions.
- Use mutation tools for focused filesystem changes.
- Use `run_command` when normal Linux commands, scripts, builds, tests, package managers, or other programs are the appropriate mechanism.
- Read the returned `ok`, `data`, and `error` information before deciding what to do next.
