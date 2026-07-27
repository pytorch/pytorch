"""
Program runner utilities for PyTorch fuzzer.
This module handles running and testing generated PyTorch programs.
"""

import os
import subprocess
import sys


def _build_subprocess_env(*, exclude_primary_device: bool = False) -> dict[str, str]:
    """Build the subprocess environment, applying the active plugin's hook (if any)."""
    from torchfuzz.codegen import get_device_info

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    select_runtime_env = get_device_info().select_runtime_env
    if select_runtime_env is not None:
        env = select_runtime_env(env, exclude_primary_device=exclude_primary_device)
    return env


class ProgramRunner:
    """Runs generated PyTorch programs and handles output/error reporting."""

    def __init__(self):
        pass

    def run_program(self, program_path):
        """
        Run a generated Python program and handle output/errors.

        Args:
            program_path: Path to the Python program to run

        Returns:
            bool: True if program ran successfully, False otherwise
        """
        abs_path = os.path.abspath(program_path)
        print(f"Running: {abs_path}")

        env = _build_subprocess_env(exclude_primary_device=True)

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            print("=== Program Output ===")
            print(result.stdout)
            print(result.stderr)
            return True

        except subprocess.CalledProcessError as e:
            print("=== Program Output (Failure) ===")
            print(e.stdout)
            print(e.stderr)
            print("===============================")
            print("=== Program Source ===")
            with open(abs_path) as f:
                print(f.read())
            print("======================")
            print(f"Program exited with code: {e.returncode}")
            sys.exit(1)
