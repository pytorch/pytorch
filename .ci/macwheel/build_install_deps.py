#!/usr/bin/env python3
"""Install build-time dependencies for the macOS arm64 wheel build.

Usage: build_install_deps.py <package_dir>

Mirrors .ci/manywheel/build_install_deps.py. macOS pins numpy by Python version
and, when the conda-forge libomp is not staged at /opt/llvm-openmp, installs
libomp from Homebrew (matching the fallback in the previous shell build).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# NumPy build-time pin per supported CPython version, mirroring the legacy
# .ci/wheel/build_wheel.sh table. Listed explicitly (rather than a prefix match
# with a default) so an unsupported version fails loudly instead of silently
# picking a fallback -- enabling a new Python here forces a deliberate pin
# choice. Freethreaded builds (e.g. 3.14t) share their base version's pin, since
# sys.version_info does not distinguish them.
NUMPY_PINS: dict[str, str] = {
    "3.10": "2.0.2",
    "3.11": "2.0.2",
    "3.12": "2.0.2",
    "3.13": "2.1.0",
    "3.14": "2.3.4",
    "3.15": "2.5.1",
}

OMP_PREFIX = Path("/opt/llvm-openmp")


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
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    pin = NUMPY_PINS.get(version)
    if pin is None:
        sys.exit(
            f"Unsupported Python version {version}: add a numpy pin to "
            "NUMPY_PINS in .ci/macwheel/build_install_deps.py"
        )
    return pin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    # requirements-build.txt supplies the build backend for `python -m build
    # --no-isolation` (the previous shell build relied on it being preinstalled).
    pip_install("-qU", "-r", "requirements-build.txt")
    pip_install("-q", "-r", "requirements.txt")
    pip_install("-q", f"numpy=={numpy_pin()}")
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
