# Sandbox quickstart

1. Install Modal and sign in:

   ```bash
   pip install modal
   gh auth login
   python -m modal setup
   ```

2. Create a sandbox from the repository root:

   ```bash
   python -m tools.sandbox create --flavor h100:1

   # Check the available flavors and pricing
   python -m tools.sandbox create --help
   ```

   GPU flavors accept counts from 1 to 8: `t4`, `l4`, `l40s`, `a100-40gb`,
   `a100-80gb`, `h100`, `h200`, `b200`, and `b300` (for example, `a100-80gb:2`).
   `a10` accepts counts from 1 to 4. `rtx-pro-6000` accepts one GPU; its
   multi-GPU limit is not documented. MI355X is not supported by Modal Sandboxes.
   New filesystems use the CUDA CI image. Reusing a filesystem preserves its
   existing toolchain and build architecture settings.

3. Connect using the printed SSH command or VS Code link.
4. Stop a sandbox to save its filesystem, then reuse the same `fs-...` ID when creating another sandbox.

   ```bash
   # Stop the sandbox
   python -m tools.sandbox stop sb-...

   # The filesystem is not deleted, so you can start another sandbox with the same filesystem
   python -m tools.sandbox create --filesystem fs-...
   ```

5. Manage sandboxes using the returned `sb-...` and `fs-...` IDs:

   | Task | Command |
   | --- | --- |
   | List sandboxes and filesystems | `python -m tools.sandbox list` |
   | Stop and save | `python -m tools.sandbox stop sb-...` |
   | Reuse a saved filesystem | `python -m tools.sandbox create --filesystem fs-... --flavor cpu` |
   | Permanently delete an unused filesystem | `python -m tools.sandbox rm fs-...` |
