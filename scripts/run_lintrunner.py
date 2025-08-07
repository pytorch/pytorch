#!/usr/bin/env python3
"""
Pre‑push hook wrapper for Lintrunner.

✓ Stores a hash of .lintrunner.toml in the venv
✓ Re-runs `lintrunner init` if that file's hash changes
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LINTRUNNER_TOML_PATH = REPO_ROOT / ".lintrunner.toml"

# This is the path to the pre-commit-managed venv
VENV_ROOT = Path(sys.executable).parent.parent
# Stores the hash of .lintrunner.toml from the last time we ran `lintrunner init`
INITIALIZED_LINTRUNNER_TOML_HASH_PATH = VENV_ROOT / ".lintrunner_plugins_hash"


def ensure_lintrunner() -> None:
    """Fail if Lintrunner is not on PATH."""
    if shutil.which("lintrunner"):
        print("✅ lintrunner is already installed")
        return
    sys.exit(
        "❌ lintrunner is required but was not found on your PATH. Please run the `python scripts/setup_hooks.py` to install to configure lintrunner before using this script. If `git push` still fails, you may need to open an new terminal"
    )


def ensure_virtual_environment() -> None:
    """Fail if not running within a virtual environment."""
    in_venv = (
        os.environ.get("VIRTUAL_ENV") is not None
        or hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

    if not in_venv:
        sys.exit(
            "❌ This script must be run from within a virtual environment. "
            "Please activate your virtual environment before running this script."
        )


def compute_file_hash(path: Path) -> str:
    """Returns SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_stored_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text().strip()
    except Exception:
        return None


def initialize_lintrunner_if_needed() -> None:
    """Runs lintrunner init if .lintrunner.toml changed since last run."""
    if not LINTRUNNER_TOML_PATH.exists():
        print("⚠️ No .lintrunner.toml found. Skipping init.")
        return

    print(
        f"INITIALIZED_LINTRUNNER_TOML_HASH_PATH = {INITIALIZED_LINTRUNNER_TOML_HASH_PATH}"
    )
    current_hash = compute_file_hash(LINTRUNNER_TOML_PATH)
    stored_hash = read_stored_hash(INITIALIZED_LINTRUNNER_TOML_HASH_PATH)

    if current_hash == stored_hash:
        print("✅ Lintrunner plugins already initialized and up to date.")
        return

    print("🔁 Running `lintrunner init` …", file=sys.stderr)
    subprocess.check_call(["lintrunner", "init"])
    INITIALIZED_LINTRUNNER_TOML_HASH_PATH.write_text(current_hash)


def main() -> None:
    # 0. Ensure we're running in a virtual environment
    ensure_virtual_environment()
    print(f"🐍 Virtual env being used: {VENV_ROOT}", file=sys.stderr)

    # 1. Ensure lintrunner binary is available
    ensure_lintrunner()

    # 2. Check for plugin updates and re-init if needed
    initialize_lintrunner_if_needed()

    # 3. Run lintrunner with any passed arguments and propagate its exit code
    args = sys.argv[1:]  # Forward all arguments to lintrunner
    result = subprocess.call(["lintrunner"] + args)
    sys.exit(result)


if __name__ == "__main__":
    main()
