import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parent.parent


def _check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ("ON", "1", "YES", "TRUE", "Y")


def _set_cmake_build_type() -> None:
    # Hotpatch CMAKE_BUILD_TYPE based on DEBUG / REL_WITH_DEB_INFO env flags so
    # CMake picks up the build type when build_libtorch.py is invoked directly
    # (the scikit-build-core path handles this via [[tool.scikit-build.overrides]]
    # in pyproject.toml). Explicit CMAKE_BUILD_TYPE wins.
    if "CMAKE_BUILD_TYPE" not in os.environ:
        if _check_env_flag("DEBUG"):
            os.environ["CMAKE_BUILD_TYPE"] = "Debug"
        elif _check_env_flag("REL_WITH_DEB_INFO"):
            os.environ["CMAKE_BUILD_TYPE"] = "RelWithDebInfo"
        else:
            os.environ["CMAKE_BUILD_TYPE"] = "Release"


def build_libtorch(rerun_cmake: bool, cmake_only: bool) -> None:
    _set_cmake_build_type()
    # Resolve the build directory relative to the current working directory,
    # not the repo root: CI invokes this script from a scratch dir to build
    # libtorch outside the source tree (see .ci/pytorch/build.sh and
    # macos-test.sh, which pushd into a temp dir first).
    build_dir = Path.cwd() / "build"
    build_dir.mkdir(exist_ok=True)

    cmake = shutil.which("cmake")
    if cmake is None:
        print("ERROR: cmake not found", file=sys.stderr)
        sys.exit(1)

    cache_file = build_dir / "CMakeCache.txt"
    if rerun_cmake:
        cache_file.unlink(missing_ok=True)
        # Drop generator state too, so a generator or toolchain change
        # actually takes effect on reconfigure.
        shutil.rmtree(build_dir / "CMakeFiles", ignore_errors=True)

    # Explicit CMAKE_GENERATOR wins; otherwise prefer ninja when available.
    generator = os.environ.get("CMAKE_GENERATOR")
    if generator is None and shutil.which("ninja"):
        generator = "Ninja"

    # Configure if needed
    if not cache_file.exists():
        args = [cmake]
        if generator:
            args += ["-G", generator]
        # Install into <repo_root>/torch so CI scripts (setup.bat) can find
        # the headers, libraries, and cmake config at torch/{include,lib,share}.
        install_prefix = REPO_ROOT / "torch"
        install_prefix.mkdir(exist_ok=True)
        args += [
            "-DBUILD_PYTHON=OFF",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            str(REPO_ROOT),
        ]
        print(" ".join(args), file=sys.stderr, flush=True)
        subprocess.check_call(args, cwd=build_dir)

    if cmake_only:
        return

    # Build. Pass --config for multi-config generators (Visual Studio on
    # Windows); single-config generators like Ninja ignore it. CMAKE_BUILD_TYPE
    # is always set by the hotpatch above.
    build_args = [
        cmake,
        "--build",
        ".",
        "--config",
        os.environ["CMAKE_BUILD_TYPE"],
        "--target",
        "install",
    ]
    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs is not None:
        build_args += ["-j", max_jobs]
    elif generator != "Ninja":
        # Ninja parallelizes by default; make and msbuild do not.
        build_args += ["-j", str(multiprocessing.cpu_count())]
    print(" ".join(build_args), file=sys.stderr, flush=True)
    subprocess.check_call(build_args, cwd=build_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build libtorch")
    parser.add_argument("--rerun-cmake", action="store_true", help="rerun cmake")
    parser.add_argument(
        "--cmake-only",
        action="store_true",
        help="Stop once cmake terminates. Leave users a chance to adjust build options",
    )
    options = parser.parse_args()
    build_libtorch(
        rerun_cmake=options.rerun_cmake,
        cmake_only=options.cmake_only,
    )
