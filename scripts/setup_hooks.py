#!/usr/bin/env python3
"""
Bootstrap Git pre‑push hook.

✓ Requires uv to be installed (fails if not available)
✓ Installs/updates pre‑commit with uv  (global, venv‑proof)
✓ Registers the repo's pre‑push hook and freezes hook versions

Run this from the repo root (inside or outside any project venv):

    python scripts/setup_hooks.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple


# ───────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────
def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_uv() -> None:
    if which("uv"):
        # Ensure the path uv installs binaries to is part of the system path
        print("$ uv tool update-shell")
        result = subprocess.run(
            ["uv", "tool", "update-shell"], capture_output=True, text=True
        )
        if result.returncode == 0:
            # Check if the output indicates changes were made
            if (
                "Updated" in result.stdout
                or "Added" in result.stdout
                or "Modified" in result.stdout
            ):
                print(
                    "⚠️  Shell configuration updated. You may need to restart your terminal for changes to take effect."
                )
            elif result.stdout.strip():
                print(result.stdout)
            return
        else:
            sys.exit(
                f"❌ Warning: uv tool update-shell failed: {result.stderr}. uv installed tools may not be available."
            )

    sys.exit(
        "\n❌  uv is required but was not found on your PATH.\n"
        "    Please install uv first using the instructions at:\n"
        "    https://docs.astral.sh/uv/getting-started/installation/\n"
        "    Then rerun  python scripts/setup_hooks.py\n"
    )


def ensure_tool_installed(
    tool: str, force_update: bool = False, python_ver: Tuple[int, int] = None
) -> None:
    """
    Checks to see if the tool is available and if not (or if force update requested) then
    it reinstalls it.

    Returns: Whether or not the tool is available on PATH.  If it's not, a new terminal
    needs to be opened before git pushes work as expected.
    """
    if force_update or not which(tool):
        print(f"Ensuring latest {tool} via uv …")
        command = ["uv", "tool", "install", "--force", tool]
        if python_ver:
            # Add the Python version to the command if specified
            command.extend(["--python", f"{python_ver[0]}.{python_ver[1]}"])
        run(command)
        if not which(tool):
            print(
                f"\n⚠️  {tool} installation succeed, but it's not on PATH. Launch a new terminal if your git pushes don't work.\n"
            )


if sys.platform.startswith("win"):
    print(
        "\n⚠️  Lintrunner is not supported on Windows, so there are no pre-push hooks to add. Exiting setup.\n"
    )
    sys.exit(0)

# ───────────────────────────────────────────
# 1. Install dependencies
# ───────────────────────────────────────────

ensure_uv()

# Ensure pre-commit is installed globally via uv
ensure_tool_installed("pre-commit", force_update=True, python_ver=(3, 9))

# Don't force a lintrunner update because it might break folks
# who already have it installed in a different way
ensure_tool_installed("lintrunner")

# ───────────────────────────────────────────
# 2. Activate (or refresh) the pre‑push hook
# ───────────────────────────────────────────

# ── Activate (or refresh) the repo’s pre‑push hook ──────────────────────────
# Creates/overwrites .git/hooks/pre‑push with a tiny shim that will call
# `pre-commit run --hook-stage pre-push` on every `git push`.
# This is why we need to install pre-commit globally.
#
# The --allow-missing-config flag lets pre-commit succeed if someone changes to
# a branch that doesn't have pre-commit installed
run(
    [
        "uv",
        "tool",
        "run",
        "pre-commit",
        "install",
        "--hook-type",
        "pre-push",
        "--allow-missing-config",
    ]
)

# ── Pin remote‑hook versions for reproducibility ────────────────────────────
# (Note: we don't have remote hooks right now, but it future-proofs this script)
# 1. `autoupdate` bumps every remote hook’s `rev:` in .pre-commit-config.yaml
#    to the latest commit on its default branch.
# 2. `--freeze` immediately rewrites each `rev:` to the exact commit SHA,
#    ensuring all contributors and CI run identical hook code.
run(["uv", "tool", "run", "pre-commit", "autoupdate", "--freeze"])


print(
    "\n✅  pre‑commit is installed globally via uv and the pre‑push hook is active.\n"
    "   Lintrunner will now run automatically on every `git push`.\n"
)
