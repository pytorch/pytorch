#!/usr/bin/env python3
"""Install build-time dependencies for a PyTorch Windows wheel build.

Windows analog of `.ci/manywheel/build_install_deps.py`. Replaces the
pip-install + libuv-extract portion of the legacy
`.ci/pytorch/windows/setup_build.bat`. The vcvarsall / CUDA / XPU env
configuration lives in the sibling `build_env_setup.py`; both scripts run
independently and hand env back to a parent bash wrapper via --env-out.

Environment variables:
    SKIP_SETUP_CLEAN - skip `spin clean` when set (build/ shared across Pythons)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# Directory containing this script (.ci/pytorch/windows). Scratch downloads
# land here so they don't pollute PYTORCH_ROOT.
WIN_CI_DIR = Path(__file__).resolve().parent
# Repo root contains pyproject.toml; spin needs to run from there.
PYTORCH_ROOT = WIN_CI_DIR.parent.parent.parent

sys.path.insert(0, str(WIN_CI_DIR))
from _common import download, write_env_exports


# Pin numpy by Python version. Matches the legacy table in setup_build.bat.
NUMPY_PINS: list[tuple[str, str]] = [
    ("cp315", "2.5.1"),
    ("cp314", "2.3.2"),
    ("cp313", "2.1.2"),
]
DEFAULT_NUMPY = "2.0.2"


# Fixed build-time pip deps from setup_build.bat. Kept hardcoded for now;
# requirements unification (gh-183913) will eventually centralize these.
PIP_PACKAGES: list[str] = [
    "cmake",
    "pyyaml",
    "mkl-include",
    "mkl-static",
    "boto3",
    "requests",
    "ninja",
    "typing_extensions",
    "setuptools==78.1.1",
    "scikit-build-core==1.0.0",
    "spin==0.17",
]


LIBUV_URL = "https://s3.amazonaws.com/ossci-windows/libuv-1.40.0-h8ffe710_0.tar.bz2"


def retry(cmd: list[str], delays: tuple[int, ...] = (1, 2, 4, 8)) -> None:
    """Run cmd, retrying with backoff on failure (mirrors the Linux helper)."""
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


def install_libuv(workdir: Path, python_prefix: Path) -> Path:
    """Curl + 7z + tar extract libuv into the running Python's prefix.

    Mirrors setup_build.bat lines 24-28. Returns libuv_ROOT.
    """
    tarball_bz2 = workdir / "libuv-1.40.0-h8ffe710_0.tar.bz2"
    tarball = workdir / "libuv-1.40.0-h8ffe710_0.tar"
    download(LIBUV_URL, tarball_bz2)
    # 7z and tar are both present on Windows CI runners (7-Zip preinstalled,
    # tar ships with Windows 10+).
    subprocess.run(["7z", "x", "-aoa", str(tarball_bz2), f"-o{workdir}"], check=True)
    python_prefix.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xvf", str(tarball), "-C", str(python_prefix)], check=True)
    libuv_root = python_prefix / "Library"
    if not libuv_root.is_dir():
        sys.exit(
            f"libuv extraction did not produce {libuv_root}; "
            "the ossci-windows tarball layout may have changed"
        )
    return libuv_root


def provide_stdalign_shim() -> None:
    """Work around a missing C11 <stdalign.h> for the cp315 numpy source build.

    numpy has no cp315 wheel on any index, so it is built from source, and
    numpy/_core/src/common/simd/simd.h includes <stdalign.h>. The Windows CI
    UCRT (Windows SDK 10.0.19041) predates that header (added ~10.0.20348), so
    MSVC fails with "C1083: Cannot open include file: 'stdalign.h'". Drop a
    minimal C11-compliant shim on INCLUDE for the cp315-only source build; in C,
    alignas/alignof are macros for the _Alignas/_Alignof operators. Remove once
    numpy ships cp315 Windows wheels or the runner SDK is >= 10.0.20348.
    """
    if sys.version_info[:2] != (3, 15):
        return
    shim_dir = WIN_CI_DIR / "stdalign_shim"
    shim_dir.mkdir(exist_ok=True)
    (shim_dir / "stdalign.h").write_text(
        "#ifndef _STDALIGN_H\n"
        "#define _STDALIGN_H\n"
        "#ifndef __cplusplus\n"
        "#define alignas _Alignas\n"
        "#define alignof _Alignof\n"
        "#endif\n"
        "#define __alignas_is_defined 1\n"
        "#define __alignof_is_defined 1\n"
        "#endif\n"
    )
    os.environ["INCLUDE"] = f"{shim_dir}{os.pathsep}{os.environ.get('INCLUDE', '')}"
    print(f"Added <stdalign.h> shim for cp315 numpy source build: {shim_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-out", type=Path)
    args = parser.parse_args()

    provide_stdalign_shim()
    pip_install("-q", f"numpy=={numpy_pin()}")
    pip_install("-q", *PIP_PACKAGES)

    if not os.environ.get("SKIP_SETUP_CLEAN"):
        subprocess.run(
            [sys.executable, "-m", "spin", "clean"], cwd=PYTORCH_ROOT, check=True
        )

    libuv_root = install_libuv(WIN_CI_DIR, Path(sys.prefix))

    write_env_exports({"libuv_ROOT": str(libuv_root)}, args.env_out)
    print(f"libuv_ROOT={libuv_root}")
    print("build_install_deps complete")


if __name__ == "__main__":
    main()
