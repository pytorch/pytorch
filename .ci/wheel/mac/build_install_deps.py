#!/usr/bin/env python3
"""Install build-time dependencies for the macOS arm64 wheel build.

Usage: build_install_deps.py <package_dir>

Mirrors .ci/wheel/linux/build_install_deps.py. macOS pins numpy by Python version
and, when the conda-forge libomp is not staged at /opt/llvm-openmp, installs
libomp from Homebrew (matching the fallback in the previous shell build).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .ci/wheel
from _common import install_numpy, pip_install, retry


OMP_PREFIX = Path("/opt/llvm-openmp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    # requirements-build.txt supplies the build backend for `python -m build
    # --no-isolation` (the previous shell build relied on it being preinstalled).
    pip_install("-qU", "-r", "requirements-build.txt")
    pip_install("-q", "-r", "requirements.txt")
    install_numpy()
    # Skip when sharing build/ across Pythons in the per-host loop -- the
    # per-Python bits (libtorch_python, _C.so) are invalidated by
    # tools/setup_helpers/cmake.py, so libtorch_cpu is reused. spin (from
    # requirements.txt above) wraps tools/clean.py and, unlike setup.py clean,
    # survives the setup.py removal in the scikit-build-core migration.
    if not os.environ.get("SKIP_SETUP_CLEAN"):
        subprocess.run([sys.executable, "-m", "spin", "clean"], check=True)

    # OpenMP: prefer the conda-forge libomp staged at /opt/llvm-openmp (set up
    # by install_libomp.sh as a separate step). Otherwise fall back to Homebrew,
    # which only supports the build machine's macOS version or higher.
    if not OMP_PREFIX.is_dir():
        if shutil.which("brew") is None:
            sys.exit("libomp not staged at /opt/llvm-openmp and brew not available")
        print("libomp not found at /opt/llvm-openmp, installing via brew")
        retry(["brew", "install", "libomp"])


if __name__ == "__main__":
    main()
