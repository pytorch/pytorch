import subprocess
import sys
import os
from typing import List


def run_cmd(cmd: List[str]) -> None:
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
    stdout, stderr = result.stdout.decode("utf-8").strip(), result.stderr.decode("utf-8").strip()
    print(stdout)
    print(stderr)
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        exit(1)


def run_timed_cmd(cmd: List[str]) -> None:
    run_cmd(["time"] + cmd)


def update_submodules() -> None:
    run_cmd(["git", "submodule", "update", "--init", "--recursive"])


def gen_compile_commands() -> None:
    os.environ["USE_NCCL"] = "0"
    os.environ["USE_DEPLOY"] = "1"
    run_timed_cmd([sys.executable, "setup.py", "--cmake-only", "build"])


def run_autogen() -> None:
    run_timed_cmd(
        [
            sys.executable,
            "-m",
            "tools.codegen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
        ]
    )

    run_timed_cmd(
        [
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--declarations-path",
            "build/aten/src/ATen/Declarations.yaml",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--nn-path",
            "aten/src",
        ]
    )


def generate_build_files() -> None:
    update_submodules()
    gen_compile_commands()
    run_autogen()
