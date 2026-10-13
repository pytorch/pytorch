# tools/sandbox

CLI for Modal dev sandboxes: `python -m tools.sandbox <command>`. Subcommands live in `cmd/<command>.py`, each exposing `add_parser(subparsers)` and `run(args)`. Shared CLI helpers live in `common.py`; filesystem ownership, saving, and deletion live in `filesystem.py`; hardware flavors and pricing live in `flavor.py`; the sandbox image definition lives in `image.py`.

A filesystem has a stable id and a named current image. Sandbox tags record the stable id and never change when saving. Hold `Filesystem.lock()` throughout create, save, or delete; also use `Filesystem.sandbox_name` for Modal's native single-owner guarantee. Save the final exit snapshot before deleting older images. Every snapshot must remain discoverable through its sandbox or named-image revision until deletion succeeds. Remove filesystem tags only after all images have been deleted and their absence verified.

## Development Best Practices

- Output:
  - Never use `print` for logs. Always use `log` from `logger.py` (`from tools.sandbox.logger import log`). It writes to stderr, so logs never mix with command output.
  - Whenever outputting results, print JSON to stdout so the output can be piped into `jq` or another command. Use `output(...)` from `common.py` (`from tools.sandbox.common import output`) rather than calling `print` or `json.dumps` yourself, so every command formats its result the same way.
  - `exec` is the one exception: it streams the remote command's own output and exits with its exit code.
- Use descriptive variable names and spell words out: `sandbox` rather than `sb`, `filesystem_exists` rather than `exists`, `sandboxes_using_filesystem` rather than `users`. Single-letter names are fine inside lambdas.
