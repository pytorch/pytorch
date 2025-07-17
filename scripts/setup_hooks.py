#!/usr/bin/env python3
"""
Bootstrap Git pre‑push hook.

✓ Installs pipx automatically (via Homebrew on macOS)
✓ Installs/updates pre‑commit with pipx  (global, venv‑proof)
✓ Registers the repo's pre‑push hook and freezes hook versions

Run this from the repo root (inside or outside any project venv):

    python scripts/setup_hooks.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


# ───────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────
def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# ───────────────────────────────────────────
# 1. Ensure pipx exists (install via brew on macOS)
# ───────────────────────────────────────────
def ensure_pipx() -> None:
    if which("pipx"):
        return
    # If we're on a mac
    if sys.platform == "darwin":
        # Try Homebrew installation
        if which("brew"):
            print("pipx not found – installing with Homebrew …")
            run(["brew", "install", "pipx"])
            run(["pipx", "ensurepath"])
        else:
            sys.exit(
                "\n❌  pipx is required but neither pipx nor Homebrew were found.\n"
                "    Please install Homebrew (https://brew.sh) or pipx manually:\n"
                "    https://pipx.pypa.io/stable/installation/\n"
            )
    else:
        # Non‑macOS: ask user to install pipx manually
        sys.exit(
            "\n❌  pipx is required but was not found on your PATH.\n"
            "    Install pipx first (https://pipx.pypa.io/stable/installation/),\n"
            "    then rerun  python scripts/setup_hooks.py\n"
        )
    if not which("pipx"):
        sys.exit(
            "\n❌  pipx installation appeared to succeed, but it's still not on PATH.\n"
            "    Restart your terminal or add pipx's bin directory to PATH and retry.\n"
        )


def ensure_tool_installed(tool: str, force_update: bool = False) -> None:
    """
    Checks to see if the tool is available and if not (or if force update requested) then
    it reinstalls it.

    Returns: Whether or not the tool is available on PATH.  If it's not, a new terminal
    needs to be opened before git pushes work as expected.
    """
    if force_update or not which(tool):
        print(f"Ensuring latest {tool} via pipx …")
        run(["pipx", "install", "--quiet", "--force", tool])
        if not which(tool):
            print(
                f"\n⚠️  {tool} installation succeed, but it's not on PATH. Launch a new terminal if your git pushes don't work.\n"
            )


if sys.platform.startswith("win"):
    print(
        "\n⚠️  Lintrunner is not supported on Windows, so there are no pre-push hooks to add. Exiting setup.\n"
    )
    sys.exit(0)

ensure_pipx()

# Ensure the path pipx installs binaries to is part of the system path.
# Modifies the shell's configuration files (like ~/.bashrc, ~/.zshrc, etc.)
#  to include the directory where pipx installs executables in your PATH
#  variable.
# Note that further down we run pre-commit through pipx because the path
#  this command adds may not be in the current shell's PATH.
run(["pipx", "ensurepath"])

# Ensure pre-commit is installed globally via pipx
ensure_tool_installed("pre-commit", force_update=True)
# Don't force a lintrunner update because it might break folks
# who already have it installed in a different way
ensure_tool_installed("lintrunner")

# ───────────────────────────────────────────
# 3. Activate (or refresh) the pre‑push hook
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
        "pipx",
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
run(["pipx", "run", "pre-commit", "autoupdate", "--freeze"])


print(
    "\n✅  pre‑commit is installed globally via pipx and the pre‑push hook is active.\n"
    "   Lintrunner will now run automatically on every `git push`.\n"
)
