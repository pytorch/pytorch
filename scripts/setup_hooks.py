#!/usr/bin/env python3
"""
Bootstrap Git pre‑push hook with isolated virtual environment.

✓ Requires uv to be installed (fails if not available)
✓ Creates isolated venv in .git/hooks/venv/ for hook dependencies
✓ Installs lintrunner only in the isolated environment
✓ Creates direct git hook that bypasses pre-commit

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
def run(cmd: list[str], cwd: Path = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_uv() -> None:
    if which("uv"):
        return

    sys.exit(
        "\n❌  uv is required but was not found on your PATH.\n"
        "    Please install uv first using the instructions at:\n"
        "    https://docs.astral.sh/uv/getting-started/installation/\n"
        "    Then rerun  python scripts/setup_hooks.py\n"
    )


def find_repo_root() -> Path:
    """Find the repository root directory and validate it's a git repo."""
    repo_root = Path(__file__).resolve().parents[1]  # Go up from scripts/ to repo root
    
    # Verify we're in a git repo
    if not (repo_root / ".git").exists():
        sys.exit(f"❌ Not a git repository. Expected .git directory at {repo_root / '.git'}")
    
    return repo_root


if sys.platform.startswith("win"):
    print(
        "\n⚠️  Lintrunner is not supported on Windows, so there are no pre-push hooks to add. Exiting setup.\n"
    )
    sys.exit(0)

# ───────────────────────────────────────────
# 1. Setup isolated hook environment
# ───────────────────────────────────────────

ensure_uv()

# Find repo root and setup hook directory
repo_root = find_repo_root()
hooks_dir = repo_root / ".git" / "hooks"
venv_dir = hooks_dir / ".venv"

print(f"Setting up isolated hook environment in {venv_dir}")

# Create isolated virtual environment for hooks
if venv_dir.exists():
    print("Removing existing hook venv...")
    shutil.rmtree(venv_dir)

run(["uv", "venv", str(venv_dir), "--python", "3.9"])

# Install lintrunner in the isolated environment
print("Installing lintrunner in isolated environment...")
run(["uv", "pip", "install", "--python", str(venv_dir / "bin" / "python"), "lintrunner"])

# ───────────────────────────────────────────
# 2. Create direct git pre-push hook
# ───────────────────────────────────────────

pre_push_hook = hooks_dir / "pre-push"

hook_script = f'''#!/bin/bash
set -e

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$HOOK_DIR/.venv"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Activate the isolated hook environment
source "$VENV_DIR/bin/activate"

# Run lintrunner wrapper
python "$REPO_ROOT/scripts/run_lintrunner.py"
'''

print(f"Creating git pre-push hook at {pre_push_hook}")
pre_push_hook.write_text(hook_script)
pre_push_hook.chmod(0o755)  # Make executable

print(
    "\n✅  Isolated hook environment created and pre‑push hook is active.\n"
    "   Lintrunner will now run automatically on every `git push`.\n"
    f"   Hook dependencies are isolated in {venv_dir}\n"
)
