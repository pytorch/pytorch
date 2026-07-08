#!/usr/bin/env python3
"""Repair (delocate) macOS arm64 wheels so they are self-contained.

Usage: repair_wheel.py <input_dir> <output_dir>

Mirrors the .ci/manywheel/repair_wheel.py contract (read raw wheels from
input_dir, write finished wheels to output_dir) but uses delocate -- the macOS
analog of auditwheel/patchelf. delocate bundles dependency dylibs into the
wheel and rewrites their install names to @loader_path. It is
interpreter-independent, so it handles any CPython ABI.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Pinned delocate release.
DELOCATE_PIN = "https://github.com/matthew-brett/delocate/archive/refs/tags/0.10.4.zip"


def ensure_delocate() -> None:
    if shutil.which("delocate-wheel") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", DELOCATE_PIN], check=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    ensure_delocate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wheels = sorted(args.input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No wheels found in {args.input_dir}")

    for whl in wheels:
        subprocess.run(
            ["delocate-wheel", "-v", "-w", str(args.output_dir), str(whl)],
            check=True,
        )

    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Delocated {len(repaired)} wheel(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
