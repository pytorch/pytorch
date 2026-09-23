#!/usr/bin/env python3
"""Smoke test for extracted libtorch: fully load libtorch.so via its $ORIGIN RPATH."""

import argparse
import ctypes
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory with the libtorch zip from extract_libtorch_from_wheel.py",
    )
    args = parser.parse_args()

    zips = [p for p in args.output_dir.glob("libtorch-*.zip") if "latest" not in p.name]
    if len(zips) != 1:
        raise RuntimeError(f"Expected exactly one libtorch zip, found: {zips}")

    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(tmp)
        libtorch = Path(tmp) / "libtorch" / "lib" / "libtorch.so"
        # Fully dlopen libtorch.so: resolves the whole NEEDED chain through the
        # $ORIGIN RPATH, so a missing or mis-rpath'd bundled dep (torch or NVIDIA
        # runtime) fails here. libcuda.so.1 is dlopen'd lazily by cudart, not a
        # load-time NEEDED, so this works on a driverless builder.
        ctypes.CDLL(str(libtorch))
        print(f"Smoke test passed: loaded {libtorch}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
