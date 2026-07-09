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


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .ci/wheel
from _common import install_numpy, pip_install


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    pip_install("-qU", "-r", "requirements-build.txt")
    # Skip when sharing build/ across Pythons in build_all.sh -- the per-Python
    # bits (libtorch_python, _C.so) are invalidated by tools/setup_helpers/cmake.py.
    if not os.environ.get("SKIP_SETUP_CLEAN"):
        subprocess.run([sys.executable, "setup.py", "clean"], check=True)
    pip_install("-q", "-r", "requirements.txt")
    install_numpy()

    if "rocm" in os.environ.get("DESIRED_CUDA", ""):
        print(f"Running build_amd.py at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        subprocess.run([sys.executable, "tools/amd_build/build_amd.py"], check=True)


if __name__ == "__main__":
    main()
