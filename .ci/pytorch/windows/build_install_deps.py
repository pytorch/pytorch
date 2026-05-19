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
import urllib.request
from pathlib import Path


# Directory containing this script (.ci/pytorch/windows). Scratch downloads
# land here so they don't pollute PYTORCH_ROOT.
WIN_CI_DIR = Path(__file__).resolve().parent


# Pin numpy by Python version. Matches the legacy table in setup_build.bat.
NUMPY_PINS: list[tuple[str, str]] = [
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
    "scikit-build-core",
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


def shell_quote(value: str) -> str:
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    if path is None:
        return
    lines = [f"export {k}={shell_quote(v)}" for k, v in env.items()]
    path.write_text("\n".join(lines) + "\n")


def download(url: str, dest: Path, attempts: int = 5) -> None:
    """Stream `url` to `dest`, retrying with backoff on failure."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{attempts})")
            with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except Exception as exc:
            if attempt == attempts:
                sys.exit(f"Failed to download {url}: {exc}")
            time.sleep(2**attempt)


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
    return python_prefix / "Library"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-out", type=Path)
    args = parser.parse_args()

    pip_install("-q", f"numpy=={numpy_pin()}")
    pip_install("-q", *PIP_PACKAGES)

    if not os.environ.get("SKIP_SETUP_CLEAN"):
        subprocess.run([sys.executable, "-m", "spin", "clean"], check=True)

    libuv_root = install_libuv(WIN_CI_DIR, Path(sys.prefix))

    write_env_exports({"libuv_ROOT": str(libuv_root)}, args.env_out)
    print(f"libuv_ROOT={libuv_root}")
    print("build_install_deps complete")


if __name__ == "__main__":
    main()
