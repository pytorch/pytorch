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
    build_dir = REPO_ROOT / "build"
    build_dir.mkdir(exist_ok=True)

    cmake = shutil.which("cmake")
    if cmake is None:
        print("ERROR: cmake not found", file=sys.stderr)
        sys.exit(1)

    cache_file = build_dir / "CMakeCache.txt"
    if rerun_cmake and cache_file.exists():
        cache_file.unlink()

    # Configure if needed
    if not cache_file.exists():
        args = [cmake]
        if shutil.which("ninja"):
            args += ["-GNinja"]
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

    # Build
    build_args = [cmake, "--build", ".", "--target", "install"]
    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs is not None:
        build_args += ["-j", max_jobs]
    elif not shutil.which("ninja"):
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
