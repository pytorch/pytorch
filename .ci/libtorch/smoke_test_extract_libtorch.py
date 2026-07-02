#!/usr/bin/env python3
"""Smoke test for extracted libtorch: verify rpath and that libtorch.so loads."""

import argparse
import shutil
import subprocess
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


def check_bundled_deps(lib_dir: Path) -> None:
    # Verify each bundled lib can resolve the dependencies that are ALSO bundled
    # in lib/. A "not found" dep whose soname is present in lib/ means its RPATH
    # was not fixed to $ORIGIN. Deps not bundled here (e.g. CUDA runtime libs)
    # are provided by the environment and are intentionally ignored.
    present = {p.name for p in lib_dir.iterdir() if p.is_file()}
    failures = []
    for so_file in sorted(lib_dir.iterdir()):
        if not so_file.is_file():
            continue
        if not (so_file.name.endswith(".so") or ".so." in so_file.name):
            continue
        result = subprocess.run(
            ["ldd", str(so_file)], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "=> not found" not in line:
                continue
            soname = line.split("=>", 1)[0].strip()
            if soname in present:
                failures.append(f"{so_file.name}: {soname} not found (RPATH not fixed)")
    if failures:
        raise RuntimeError("Bundled dependency resolution failed:\n" + "\n".join(failures))
    print("  bundled deps OK")


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
            raise FileNotFoundError("libtorch/lib not found in zip")
        check_rpath(lib_dir)
        check_bundled_deps(lib_dir)
        print("Smoke test passed.")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
