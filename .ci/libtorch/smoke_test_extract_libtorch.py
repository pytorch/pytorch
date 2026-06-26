#!/usr/bin/env python3
"""Smoke test for extracted libtorch: verify rpath and that libtorch.so loads."""

import argparse
import ctypes
import glob
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check_rpath(lib_dir: Path) -> None:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        print("patchelf not found, skipping rpath checks")
        return
    for so_file in sorted(lib_dir.iterdir()):
        if not so_file.is_file():
            continue
        if not (so_file.name.endswith(".so") or ".so." in so_file.name):
            continue
        result = subprocess.run(
            [patchelf, "--print-rpath", str(so_file)],
            capture_output=True,
            text=True,
        )
        rpath = result.stdout.strip()
        if "$ORIGIN" not in rpath:
            raise RuntimeError(
                f"{so_file.name}: expected $ORIGIN in rpath, got {rpath!r}"
            )
        print(f"  rpath OK: {so_file.name} -> {rpath}")


def check_import(lib_dir: Path) -> None:
    libtorch = lib_dir / "libtorch.so"
    if not libtorch.exists():
        raise FileNotFoundError(f"libtorch.so not found in {lib_dir}")
    ctypes.CDLL(str(libtorch))
    print(f"  import OK: loaded {libtorch.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory containing the libtorch zip produced by extract_libtorch_from_wheel.py",
    )
    args = parser.parse_args()

    zips = [
        p for p in args.output_dir.glob("libtorch-*.zip")
        if "latest" not in p.name
    ]
    if not zips:
        raise FileNotFoundError(f"No libtorch zip found in {args.output_dir}")
    if len(zips) > 1:
        raise RuntimeError(f"Multiple libtorch zips found: {zips}")
    libtorch_zip = zips[0]

    tmp = tempfile.mkdtemp()
    try:
        print(f"Extracting {libtorch_zip.name} ...")
        with zipfile.ZipFile(libtorch_zip) as zf:
            zf.extractall(tmp)
        lib_dir = Path(tmp) / "libtorch" / "lib"
        if not lib_dir.is_dir():
            raise FileNotFoundError(f"libtorch/lib not found in zip")
        check_rpath(lib_dir)
        check_import(lib_dir)
        print("Smoke test passed.")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
