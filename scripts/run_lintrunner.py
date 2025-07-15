#!/usr/bin/env python3
"""
Pre‑push hook wrapper for Lintrunner.

✓ Skips entirely on CI (when $CI is set)
✓ Installs Lintrunner once (`pip install lintrunner`) if missing
✓ Runs Lintrunner and propagates its exit status
✓ Pure Python – works on macOS, Linux, Windows
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys

def ensure_lintrunner() -> None:
    """Install Lintrunner globally (user site) if not already on PATH."""
    if shutil.which("lintrunner"):
        return
    print("Installing lintrunner …", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--quiet", "lintrunner"])
    subprocess.check_call(["lintrunner", "init"])

def main() -> None:
    
    # 1) Skip the hook entirely on CI runners
    if os.environ.get("CI"):
        sys.exit(0)

    import sys
    print("Lintrunner is using Python:", sys.executable, file=sys.stderr)
    
    # 2) Ensure lintrunner is available, then run it
    ensure_lintrunner()
    result = subprocess.call(["lintrunner"])
    sys.exit(result)

if __name__ == "__main__":
    main()
