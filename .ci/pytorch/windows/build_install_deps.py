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
    ("cp315", "2.5.2"),
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


def preinstall_cp315_build_deps() -> list[str]:
    """Pin Cython for the cp315 sdist builds and return the pip flags to use.

    Cython 3.3.0 (released 2026-08-22 05:16 UTC) crashes with
    STATUS_ACCESS_VIOLATION (0xC0000005, decimal 3221225477) under the
    GIL-enabled CPython 3.15 build on Windows, so every package pip has to
    build from an sdist dies in its PEP 517 hook. meson projects surface it as
    the misleading "Unknown compiler(s): [['cython'], ['cython3']]"; setuptools
    ones just exit 3221225477 with no output at all.

    The Windows nightly passed on 2026-08-21 and has failed every run since
    2026-08-22, the first after that release, with no PyTorch-side change and
    no setuptools/wheel/packaging release in the window. cp315t is unaffected,
    so the crash is specific to the GIL-enabled build.

    PIP_CONSTRAINT does not work here: pip does not apply it to the isolated
    build environments, which is where the crashing Cython is installed. That
    was verified directly with pip 26.2.1 -- building pyyaml from its sdist
    installs Cython 3.3.0 into the build env whether or not PIP_CONSTRAINT
    pins it lower. So instead pin Cython in the outer environment and turn
    build isolation off, which makes sdist builds use the pinned copy.

    Remove once Cython ships a fix.
    """
    if sys.version_info[:2] != (3, 15):
        return []
    # setuptools/wheel must be present too: without isolation pip will not
    # provide them to the sdist builds.
    pip_install("-q", "cython<3.3.0", "setuptools", "wheel")
    print("Pinned cython<3.3.0 for cp315 sdist builds; disabling build isolation")
    return ["--no-build-isolation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-out", type=Path)
    args = parser.parse_args()

    build_flags = preinstall_cp315_build_deps()
    pip_install("-q", *build_flags, f"numpy=={numpy_pin()}")
    pip_install("-q", *build_flags, *PIP_PACKAGES)

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
