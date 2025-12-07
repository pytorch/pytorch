"""
Program runner utilities for PyTorch fuzzer.
This module handles running and testing generated PyTorch programs.
"""

import os
import random
import subprocess
import sys


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

        # Select a random CUDA device if available
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        else:
            # Default to all GPUs if not set
            try:
                import torch

                num_gpus = torch.cuda.device_count()
                devices = [str(i) for i in range(num_gpus)]
            except ImportError:
                devices = []
        if devices:
            selected_device = random.choice(devices)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = selected_device
            print(f"Selected CUDA_VISIBLE_DEVICES={selected_device}")
        else:
            env = None  # No GPU available or torch not installed

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

    def run_and_validate(self, program_path):
        """
        Run a program and return detailed results for validation.

        Args:
            program_path: Path to the Python program to run

        Returns:
            dict: Dictionary with 'success', 'stdout', 'stderr', 'returncode'
        """
        abs_path = os.path.abspath(program_path)

        # Select a random CUDA device if available
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        else:
            try:
                import torch

                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    devices = [str(i) for i in range(1, num_gpus)]
                else:
                    devices = [str(i) for i in range(num_gpus)]
            except ImportError:
                devices = []
        if devices:
            selected_device = random.choice(devices)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = selected_device
            print(f"Selected CUDA_VISIBLE_DEVICES={selected_device}")
        else:
            env = None

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode,
            }
