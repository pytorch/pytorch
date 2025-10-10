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


def gen_compile_commands() -> None:
    os.environ["USE_NCCL"] = "0"
    os.environ["USE_PRECOMPILED_HEADERS"] = "1"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    os.environ["CMAKE_ONLY"] = "1"
    run_cmd([sys.executable, "setup.py", "build"])


def run_autogen() -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
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
