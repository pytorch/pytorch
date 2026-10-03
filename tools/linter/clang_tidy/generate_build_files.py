from __future__ import annotations

import os
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    print(stdout)
    print(stderr)
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        sys.exit(1)


def update_submodules() -> None:
    run_cmd(["git", "submodule", "update", "--init", "--recursive"])


# Deliberately not the `build` directory used by `pip install -e .` (see
# `build-dir` in pyproject.toml): CMake takes CC/CXX and the USE_* cache
# variables below from the environment only when it creates the cache, so
# reusing the developer's build directory would silently keep their compiler
# and overwrite their cache variables. Must match the `--build_dir` passed to
# the CLANGTIDY linters in .lintrunner.toml.
BUILD_DIR = "build_lint"


def gen_compile_commands() -> None:
    """Configure cmake to produce compile_commands.json for clang-tidy.

    Configure-only invocation; does not run the build step. The repo-level
    cmake/EnvVarForwarding.cmake forwards BUILD_*/USE_* environment
    variables to the corresponding CMake cache variables, so setting them
    in os.environ before this call propagates them through to CMake.
    """
    os.environ["USE_NCCL"] = "0"
    os.environ["USE_PRECOMPILED_HEADERS"] = "1"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    run_cmd(["cmake", "-S", ".", "-B", BUILD_DIR, "-G", "Ninja"])


def run_autogen() -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            f"{BUILD_DIR}/aten/src/ATen",
            "--per-operator-headers",
        ]
    )

    run_cmd(
        [
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--gen-lazy-ts-backend",
        ]
    )


def generate_build_files() -> None:
    update_submodules()
    gen_compile_commands()
    run_autogen()


if __name__ == "__main__":
    generate_build_files()
