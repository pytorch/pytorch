#!/usr/bin/env python3
"""
Pre‑push hook wrapper for Lintrunner.

✓ Skips entirely on CI (when $CI is set)
✓ Installs Lintrunner once (`pip install lintrunner`) if missing
✓ Stores a hash of .lintrunner.toml in the venv
✓ Re-runs `lintrunner init` if that file's hash changes
✓ Pure Python – works on macOS, Linux, and Windows
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOML_PATH = REPO_ROOT / ".lintrunner.toml"

# This is the path to the pre-commit-managed venv
VENV_ROOT = Path(sys.executable).parent.parent
MARKER_PATH = VENV_ROOT / ".lintrunner_plugins_hash"


def ensure_lintrunner() -> None:
    """Fail if Lintrunner is not on PATH."""
    if shutil.which("lintrunner"):
        print("✅ lintrunner is already installed")
        return
    sys.exit("❌ lintrunner is required but was not found on your PATH. Please install it via `pipx install lintrunner` before running this script.")


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


def maybe_initialize_lintrunner() -> None:
    """Runs lintrunner init if .lintrunner.toml changed since last run."""
    if not TOML_PATH.exists():
        print("⚠️ No .lintrunner.toml found. Skipping init.")
        return

    current_hash = compute_file_hash(TOML_PATH)
    stored_hash = read_stored_hash(MARKER_PATH)

    if current_hash == stored_hash:
        print("✅ Lintrunner plugins already initialized and up to date.")
        return

    print("🔁 Running `lintrunner init` …", file=sys.stderr)
    subprocess.check_call(["lintrunner", "init"])
    MARKER_PATH.write_text(current_hash)


def main() -> None:
    # 1. Skip in CI
    if os.environ.get("CI"):
        print("⚙️ CI detected — skipping lintrunner")
        sys.exit(0)

    print(f"🐍 Lintrunner is using Python: {sys.executable}", file=sys.stderr)

    # 2. Ensure lintrunner binary is available
    ensure_lintrunner()

    # 3. Check for plugin updates and re-init if needed
    maybe_initialize_lintrunner()

    # 4. Run lintrunner with any passed arguments and propagate its exit code
    args = sys.argv[1:]  # Forward all arguments to lintrunner
    result = subprocess.call(["lintrunner"] + args)
    sys.exit(result)


if __name__ == "__main__":
    main()
