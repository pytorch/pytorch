#!/usr/bin/env python3
# Owner(s): ["module: openreg"]

"""
OpenReg Backend Tests

This test module validates PyTorch's third-party accelerator integration mechanism
by running tests for the torch_openreg backend. The torch_openreg backend is similar
to CUDA or MPS and demonstrates the open registration extension pattern.

If all tests pass, it indicates that the third-party accelerator integration
mechanism is working as expected.
"""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parent.parent


def install_cpp_extensions(extensions_dir):
    """
    Install C++ extensions for testing.

    Args:
        extensions_dir: Directory containing the C++ extension to build

    Returns:
        Tuple of (install_directory, return_code)
    """
    # Wipe the build folder, if it exists already
    build_dir = os.path.join(extensions_dir, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # Build the test cpp extensions modules
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        ".",
        "--root",
        "./install",
    ]

    result = subprocess.run(
        cmd,
        cwd=extensions_dir,
        env=os.environ,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Failed to install C++ extensions:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None, result.returncode

    # Get the site-packages directory prepared for PYTHONPATH
    platlib_path = sysconfig.get_paths()["platlib"]
    platlib_rel = os.path.relpath(
        platlib_path, os.path.splitdrive(platlib_path)[0] + os.sep
    )
    install_directory = os.path.join(extensions_dir, "install", platlib_rel)

    assert install_directory, "install_directory must not be empty"
    return install_directory, 0


class TestOpenReg(TestCase):
    """
    Test case for OpenReg backend functionality.

    This test builds and installs the torch_openreg C++ extension and runs
    all tests contained within it using unittest discovery.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the OpenReg extension before running tests."""
        super().setUpClass()

        # Locate the torch_openreg directory
        test_directory = REPO_ROOT / "test"
        cls.openreg_dir = (
            test_directory
            / "cpp_extensions"
            / "open_registration_extension"
            / "torch_openreg"
        )

        # Install the C++ extension
        cls.install_dir, return_code = install_cpp_extensions(str(cls.openreg_dir))

        if return_code != 0:
            raise RuntimeError(
                f"Failed to install torch_openreg extension (exit code: {return_code})"
            )

        # Add to PYTHONPATH for the test session
        cls.original_python_path = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [cls.install_dir, cls.original_python_path]
        )

    @classmethod
    def tearDownClass(cls):
        """Restore original PYTHONPATH after tests."""
        os.environ["PYTHONPATH"] = cls.original_python_path
        super().tearDownClass()

    def test_openreg_extension(self):
        """
        Run all OpenReg extension tests using unittest discovery.

        This test discovers and runs all tests in the torch_openreg/tests directory.
        """
        test_directory = REPO_ROOT / "test"
        tests_dir = self.openreg_dir / "tests"

        # Use unittest discovery to find and run all tests
        cmd = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(tests_dir),
            "-v",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(test_directory),
            env=os.environ,
            capture_output=True,
            text=True,
        )

        # Print output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Assert that the tests passed
        self.assertEqual(
            result.returncode,
            0,
            f"OpenReg extension tests failed with exit code {result.returncode}",
        )


if __name__ == "__main__":
    run_tests()
