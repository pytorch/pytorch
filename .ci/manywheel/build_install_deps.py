#!/usr/bin/env python3
"""Install build-time dependencies for a PyTorch wheel build.

Usage: build_install_deps.py <package_dir>

Environment variables:
    DESIRED_CUDA - CUDA variant; "rocm*" triggers the AMD source-rewrite step.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# NumPy build-time pin selected by Python version.
# Keep in sync with .ci/manywheel/build_common.sh.
NUMPY_PINS: list[tuple[str, str]] = [
    ("cp314", "2.3.4"),
    ("cp31", "2.1.0"),
]
DEFAULT_NUMPY = "2.0.2"


def retry(cmd: list[str], delays: tuple[int, ...] = (1, 2, 4, 8)) -> None:
    """Run cmd, retrying with backoff on failure (mirrors the shell retry helper)."""
    last_rc = 0
    for delay in (0, *delays):
        if delay:
            time.sleep(delay)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        last_rc = result.returncode
    sys.exit(last_rc)


def pip_install(*args: str) -> None:
    retry([sys.executable, "-m", "pip", "install", *args])


def numpy_pin() -> str:
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for prefix, version in NUMPY_PINS:
        if tag.startswith(prefix):
            return version
    return DEFAULT_NUMPY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    pip_install("-qU", "-r", "requirements-build.txt")
    subprocess.run([sys.executable, "setup.py", "clean"], check=True)
    pip_install("-q", "-r", "requirements.txt")
    pip_install("-q", "--pre", f"numpy=={numpy_pin()}")

    if "rocm" in os.environ.get("DESIRED_CUDA", ""):
        print(f"Running build_amd.py at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        subprocess.run([sys.executable, "tools/amd_build/build_amd.py"], check=True)


if __name__ == "__main__":
    main()
