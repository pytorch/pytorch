#!/usr/bin/env python3
"""Embed macOS OpenMP into the PyTorch wheel.

Invoked at CMake install time. Copies libomp.dylib next to libtorch_cpu.dylib
and rewrites libtorch_cpu's load command / rpath so the bundled copy is the
sole resolution path at runtime.

Handles two build environments:
  1. Homebrew-style:  LC_LOAD_DYLIB = "<abs>/libomp.dylib"
     -> change to "@rpath/libomp.dylib", add @loader_path rpath
  2. Conda-style:     LC_LOAD_DYLIB = "@rpath/libomp.dylib", LC_RPATH = "<conda>/lib"
     -> replace the conda rpath with @loader_path (load cmd untouched)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


_LOAD_RE = re.compile(r"(?:name|path) (.+) \(offset \d+\)")


def parse_otool(binary: Path) -> tuple[list[str], list[str]]:
    """Return (rpaths, load_libs) extracted from `otool -l`."""
    out = subprocess.check_output(["otool", "-l", str(binary)], text=True).splitlines()
    rpaths: list[str] = []
    libs: list[str] = []
    for i, line in enumerate(out):
        s = line.strip()
        if s == "cmd LC_RPATH":
            m = _LOAD_RE.match(out[i + 2].strip())
            if m:
                rpaths.append(m.group(1))
        elif s == "cmd LC_LOAD_DYLIB":
            m = _LOAD_RE.match(out[i + 2].strip())
            if m:
                libs.append(m.group(1))
    return rpaths, libs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--libomp-path",
        required=True,
        help="Absolute path to libomp.dylib (CMake's OpenMP_libomp_LIBRARY)",
    )
    p.add_argument(
        "--lib-dir",
        required=True,
        help="Install destination dir for libomp.dylib (e.g. <install>/lib)",
    )
    args = p.parse_args()

    libomp_path = Path(args.libomp_path)
    lib_dir = Path(args.lib_dir)
    libtorch_cpu = lib_dir / "libtorch_cpu.dylib"

    if not libtorch_cpu.exists() or not libomp_path.exists():
        return 0

    libomp_name = libomp_path.name
    rpath_ref = f"@rpath/{libomp_name}"
    rpaths, libs = parse_otool(libtorch_cpu)

    # Skip if libtorch_cpu doesn't actually link libomp.
    if str(libomp_path) not in libs and rpath_ref not in libs:
        return 0

    target = lib_dir / libomp_name
    if not target.exists():
        shutil.copy2(libomp_path, target)

    # Homebrew/abs-path case: rewrite the load command to @rpath form.
    if str(libomp_path) in libs:
        subprocess.check_call(
            [
                "install_name_tool",
                "-change",
                str(libomp_path),
                rpath_ref,
                str(libtorch_cpu),
            ]
        )

    # Replace any rpath pointing at the build-time libomp directory so dyld
    # cannot fall back to it at runtime.
    libomp_dir = libomp_path.parent
    for rp in rpaths:
        if Path(rp) == libomp_dir:
            subprocess.check_call(
                [
                    "install_name_tool",
                    "-rpath",
                    rp,
                    "@loader_path",
                    str(libtorch_cpu),
                ]
            )
            return 0

    # No matching rpath -- add @loader_path if not already present.
    if "@loader_path" not in rpaths:
        subprocess.check_call(
            [
                "install_name_tool",
                "-add_rpath",
                "@loader_path",
                str(libtorch_cpu),
            ]
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
