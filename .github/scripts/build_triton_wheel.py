#!/usr/bin/env python3

import os
import re
import shutil
import sys
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Optional


SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent


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
    path: Path, *, version: str, expected_version: Optional[str] = None
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

def get_rocm_version() -> str:
    rocm_path = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH') or "/opt/rocm"
    rocm_version = "0.0.0"
    rocm_version_h = f"{rocm_path}/include/rocm-core/rocm_version.h"
    if not os.path.isfile(rocm_version_h):
        rocm_version_h = f"{rocm_path}/include/rocm_version.h"
    # The file could be missing due to 1) ROCm version < 5.2, or 2) no ROCm install.
    if os.path.isfile(rocm_version_h):
        RE_MAJOR = re.compile(r"#define\s+ROCM_VERSION_MAJOR\s+(\d+)")
        RE_MINOR = re.compile(r"#define\s+ROCM_VERSION_MINOR\s+(\d+)")
        RE_PATCH = re.compile(r"#define\s+ROCM_VERSION_PATCH\s+(\d+)")
        major, minor, patch = 0, 0, 0
        for line in open(rocm_version_h):
            match = RE_MAJOR.search(line)
            if match:
                major = int(match.group(1))
            match = RE_MINOR.search(line)
            if match:
                minor = int(match.group(1))
            match = RE_PATCH.search(line)
            if match:
                patch = int(match.group(1))
        rocm_version = str(major)+"."+str(minor)+"."+str(patch)
    return rocm_version

def build_triton(
    *,
    version: str,
    commit_hash: str,
    device: str = "cuda",
    py_version: Optional[str] = None,
    release: bool = False,
    with_clang_ldd: bool = False,
) -> Path:
    env = os.environ.copy()
    if "MAX_JOBS" not in env:
        max_jobs = os.cpu_count() or 1
        env["MAX_JOBS"] = str(max_jobs)
    if not release:
        # Nightly binaries include the triton commit hash, i.e. 2.1.0+e6216047b8
        # while release build should only include the version, i.e. 2.1.0
        rocm_version = get_rocm_version()
        version_suffix = f"+rocm{rocm_version}.git{commit_hash[:8]}"
        version += version_suffix
    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"

        triton_repo = "https://github.com/openai/triton"
        if device == "rocm":
            triton_pkg_name = "pytorch-triton-rocm"
        elif device == "xpu":
            triton_pkg_name = "pytorch-triton-xpu"
            triton_repo = "https://github.com/intel/intel-xpu-backend-for-triton"
        else:
            triton_pkg_name = "pytorch-triton"
        check_call(["git", "clone", triton_repo, "triton"], cwd=tmpdir)
        if release:
            ver, rev, patch = version.split(".")
            check_call(
                ["git", "checkout", f"release/{ver}.{rev}.x"], cwd=triton_basedir
            )
        else:
            check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

        # change built wheel name and version
        env["TRITON_WHEEL_NAME"] = triton_pkg_name
        env["TRITON_WHEEL_VERSION_SUFFIX"] = version_suffix
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
