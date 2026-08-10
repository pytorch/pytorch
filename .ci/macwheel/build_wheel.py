#!/usr/bin/env python3
"""Build a macOS arm64 PyTorch wheel.

Usage: build_wheel.py <output_dir>

Mirrors .ci/manywheel/build_wheel.py. Build flags are expected in the
environment (set by build_env_setup.py and sourced by build.sh); this script
adds only the macOS host-platform plumbing and runs `python -m build`.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Force the wheel's platform tag and compile arch to arm64 regardless of the
# build host, via `_PYTHON_HOST_PLATFORM` / `ARCHFLAGS`.
MACOS_PLATFORM = "macosx-14.0-arm64"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("_PYTHON_HOST_PLATFORM", MACOS_PLATFORM)
    env.setdefault("ARCHFLAGS", "-arch arm64")

    subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(args.output_dir),
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
