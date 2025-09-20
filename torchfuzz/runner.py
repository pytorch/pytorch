"""
Program runner utilities for PyTorch fuzzer.
This module handles running and testing generated PyTorch programs.
"""

import subprocess
import sys
import os


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
        print("Running generated program...")

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True
            )
            print("=== Program Output ===")
            print(result.stdout)
            print("======================")
            return True

        except subprocess.CalledProcessError as e:
            print("=== Program Output (Failure) ===")
            print(e.stdout)
            print(e.stderr)
            print("===============================")
            print("=== Program Source ===")
            with open(abs_path, "r") as f:
                print(f.read())
            print("======================")
            print(f"Program exited with code: {e.returncode}")
            return False

    def run_and_validate(self, program_path):
        """
        Run a program and return detailed results for validation.

        Args:
            program_path: Path to the Python program to run

        Returns:
            dict: Dictionary with 'success', 'stdout', 'stderr', 'returncode'
        """
        abs_path = os.path.abspath(program_path)

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True
            )
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'returncode': e.returncode
            }
