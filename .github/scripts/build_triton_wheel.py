#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import http.client
import io
import os
import platform
import shutil
import sys
import tarfile
import time
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent

_PTXAS_13_4_46_TRITON_PIN = "ef4ab63bf41fc21e63bf3d77d11d9365837d0254"
_PTXAS_13_4_46_PACKAGES = {
    "x86_64": (
        "amd64",
        "x86_64",
        "4664ae5f28e4eaebf8fea98eca879299a71ee9e54943a5c5a30774f18b69b44e",
    ),
    "aarch64": (
        "arm64",
        "sbsa",
        "88cfe8bee7b12d380a05286545462be1de9c6f303ee9bef2a045b3f06ad2fe4e",
    ),
}


def _read_ar_member(archive: bytes, wanted_name: str) -> bytes:
    """Read one member from the simple ar container used by Debian packages."""
    if not archive.startswith(b"!<arch>\n"):
        raise RuntimeError("The ptxas preview package is not a Debian archive")

    offset = 8
    while offset < len(archive):
        header = archive[offset : offset + 60]
        if len(header) != 60 or header[58:60] != b"`\n":
            raise RuntimeError("The ptxas preview package has an invalid ar header")
        name = header[:16].decode("ascii").strip().rstrip("/")
        size = int(header[48:58].decode("ascii").strip())
        start = offset + 60
        end = start + size
        if end > len(archive):
            raise RuntimeError("The ptxas preview package has a truncated ar member")
        if name == wanted_name:
            return archive[start:end]
        offset = end + size % 2

    raise RuntimeError(f"Can't find {wanted_name} in the ptxas preview package")


def _download_verified(url: str, expected_sha256: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urlopen(url, timeout=60) as response:
                contents = response.read()
            actual_sha256 = hashlib.sha256(contents).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"checksum mismatch: expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )
            return contents
        except (http.client.HTTPException, OSError, RuntimeError) as error:
            last_error = error
            if attempt < 5:
                print(f"Download attempt {attempt} failed: {error}; retrying")
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Failed to download {url} after 5 attempts") from last_error


def seed_preview_ptxas(commit_hash: str) -> None:
    """Seed Triton's cache when testing the ptxas 13.4.46 preview pin."""
    if commit_hash != _PTXAS_13_4_46_TRITON_PIN:
        return

    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if machine not in _PTXAS_13_4_46_PACKAGES:
        raise RuntimeError(f"Unsupported architecture for ptxas 13.4.46: {machine}")
    package_arch, triton_arch, expected_sha256 = _PTXAS_13_4_46_PACKAGES[machine]
    package = f"cuda-nvcc-13-4_13.4.46-1_{package_arch}.deb"
    package_url = (
        f"https://packages.nvidia.com/jammy/pool/{package_arch}/"
        f"5B515474-7E78-11F1-8656-C51E4F4B317F/{package}"
    )

    print(f"Downloading ptxas 13.4.46 preview package from {package_url}")
    package_contents = _download_verified(package_url, expected_sha256)

    data_archive = _read_ar_member(package_contents, "data.tar.xz")
    with tarfile.open(fileobj=io.BytesIO(data_archive), mode="r:xz") as archive:
        ptxas = archive.extractfile("./usr/local/cuda-13.4/bin/ptxas")
        if ptxas is None:
            raise RuntimeError("Can't extract ptxas from the preview package")
        triton_home = Path(os.environ.get("TRITON_HOME", Path.home()))
        cache_path = (
            triton_home
            / ".triton"
            / "nvidia"
            / "nvcc-blackwell"
            / f"cuda_nvcc-linux-{triton_arch}-13.4.46-archive"
            / "bin"
            / "ptxas"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(ptxas.read())
        cache_path.chmod(0o755)
        print(f"Seeded Triton ptxas cache at {cache_path}")


def read_triton_pin(device: str = "cuda") -> str:
    triton_file = "triton.txt"
    if device == "xpu":
        triton_file = "triton-xpu.txt"
    with open(REPO_DIR / ".ci" / "docker" / "ci_commit_pins" / triton_file) as f:
        return f.read().strip()


def read_triton_version(device: str = "cuda") -> str:
    triton_version_file = "triton_version.txt"
    if device == "xpu":
        triton_version_file = "triton_xpu_version.txt"
    with open(REPO_DIR / ".ci" / "docker" / triton_version_file) as f:
        return f.read().strip()


def check_and_replace(inp: str, src: str, dst: str) -> str:
    """Checks that `src` can be found in `input` and replaces it with `dst`"""
    if src not in inp:
        raise RuntimeError(f"Can't find ${src} in the input")
    return inp.replace(src, dst)


def patch_init_py(
    path: Path, *, version: str, expected_version: str | None = None
) -> None:
    if not expected_version:
        expected_version = read_triton_version()
    with open(path) as f:
        orig = f.read()
    # Replace version
    orig = check_and_replace(
        orig, f"__version__ = '{expected_version}'", f'__version__ = "{version}"'
    )
    with open(path, "w") as f:
        f.write(orig)


def build_triton(
    *,
    version: str,
    commit_hash: str,
    device: str = "cuda",
    py_version: str | None = None,
    release: bool = False,
    with_clang_ldd: bool = False,
) -> Path:
    env = os.environ.copy()
    # Default to the CPUs available to this process: os.sched_getaffinity respects
    # cgroup/cpuset limits (e.g. inside an OSDC container), unlike os.cpu_count()
    # which reports the whole host. A pre-set MAX_JOBS still wins.
    if hasattr(os, "sched_getaffinity"):
        default_jobs = len(os.sched_getaffinity(0))
    else:
        default_jobs = os.cpu_count() or 1
    max_jobs = int(env.get("MAX_JOBS", default_jobs))
    env["MAX_JOBS"] = str(max_jobs)

    if device == "xpu" and "TRITON_PARALLEL_LINK_JOBS" not in env:
        env["TRITON_PARALLEL_LINK_JOBS"] = str(max_jobs // 2 or 1)

    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"

        triton_repo = "https://github.com/openai/triton"
        if device == "rocm":
            triton_pkg_name = "triton-rocm"
        elif device == "xpu":
            triton_pkg_name = "triton-xpu"
            triton_repo = "https://github.com/intel/intel-xpu-backend-for-triton"
        else:
            triton_pkg_name = "triton"
        check_call(["git", "clone", triton_repo, "triton"], cwd=tmpdir)
        if release:
            ver, rev, patch = version.split(".")
            if device == "xpu":
                # XPU uses the patch version in the release branch name
                check_call(
                    ["git", "checkout", f"release/{ver}.{rev}.{patch}"],
                    cwd=triton_basedir,
                )
            else:
                check_call(
                    ["git", "checkout", f"release/{ver}.{rev}.x"], cwd=triton_basedir
                )
        else:
            check_call(["git", "fetch", "origin", commit_hash], cwd=triton_basedir)
            check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

        seed_preview_ptxas(commit_hash)

        # change built wheel name and version
        env["TRITON_WHEEL_NAME"] = triton_pkg_name
        if sys.platform != "win32":
            env["TRITON_EXT_ENABLED"] = "ON"
        if with_clang_ldd:
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "1"

        patch_init_py(
            triton_pythondir / "triton" / "__init__.py",
            version=f"{version}",
            expected_version=read_triton_version(device),
        )

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/package_triton_wheel.sh"],
                cwd=triton_basedir,
                shell=True,
            )
            print("ROCm libraries setup for triton installation...")

        # old triton versions have setup.py in the python/ dir,
        # new versions have it in the root dir.
        triton_setupdir = (
            triton_basedir
            if (triton_basedir / "setup.py").exists()
            else triton_pythondir
        )

        check_call(
            [sys.executable, "setup.py", "bdist_wheel"], cwd=triton_setupdir, env=env
        )

        whl_path = next(iter((triton_setupdir / "dist").glob("*.whl")))
        shutil.copy(whl_path, Path.cwd())

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/patch_triton_wheel.sh", Path.cwd()],
                cwd=triton_basedir,
            )

        return Path.cwd() / whl_path.name


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Build Triton binaries")
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "rocm", "xpu", "aarch64"]
    )
    parser.add_argument("--py-version", type=str)
    parser.add_argument("--commit-hash", type=str)
    parser.add_argument("--with-clang-ldd", action="store_true")
    parser.add_argument("--triton-version", type=str, default=None)
    args = parser.parse_args()

    triton_version = read_triton_version(args.device)
    if args.triton_version:
        triton_version = args.triton_version

    build_triton(
        device=args.device,
        commit_hash=(
            args.commit_hash if args.commit_hash else read_triton_pin(args.device)
        ),
        version=triton_version,
        py_version=args.py_version,
        release=args.release,
        with_clang_ldd=args.with_clang_ldd,
    )


if __name__ == "__main__":
    main()
