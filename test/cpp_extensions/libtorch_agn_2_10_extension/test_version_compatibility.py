# Owner(s): ["module: cpp"]

"""
Unit tests to verify that each function file requires PyTorch 2.10+.

This test suite compiles each .cpp file in the csrc directory with
TORCH_TARGET_VERSION=2.9.0 and expects compilation to fail.
If compilation succeeds, it means that either

(1) The test function works with 2.9.0 and should not be in this directory.
(2) The test function tests APIs that do not have proper TORCH_FEATURE_VERSION
    guards. If this is the case, and you incorrectly move the test function into
    libtorch_agn_2_9_extension the libtorch_agnostic_targetting CI workflow
    will catch this.

Run this script with VERSION_COMPAT_DEBUG=1 to see compilation errors.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase
from torch.utils.cpp_extension import (
    CUDA_HOME,
    include_paths as torch_include_paths,
    ROCM_HOME,
)


GPU_HOME = CUDA_HOME or ROCM_HOME

# TODO: Fix this error in Windows:
# numba.cuda.cudadrv.driver:driver.py:384 Call to cuInit results in CUDA_ERROR_NO_DEVICE
if not IS_WINDOWS:

    class FunctionVersionCompatibilityTest(TestCase):
        """Test that all function files require PyTorch 2.10+."""

        @classmethod
        def setUpClass(cls):
            """Set up test environment once for all tests."""
            cls.csrc_dir = Path(__file__).parent / "libtorch_agn_2_10" / "csrc"
            cls.build_dir = Path(tempfile.mkdtemp(prefix="version_check_"))

            cls.pytorch_includes = [
                f"-I{path}" for path in torch_include_paths(device_type="cpu")
            ]
            cls.cuda_includes = []
            if GPU_HOME:
                cuda_include_path = os.path.join(GPU_HOME, "include")
                if os.path.exists(cuda_include_path):
                    cls.cuda_includes = [f"-I{cuda_include_path}"]

            cls.cuda_available = cls._check_cuda_available()

        @classmethod
        def tearDownClass(cls):
            """Clean up build directory."""
            import shutil

            if cls.build_dir.exists():
                shutil.rmtree(cls.build_dir)

        @staticmethod
        def _check_cuda_available() -> bool:
            """Check if CUDA is available."""
            try:
                import torch

                return torch.cuda.is_available()
            except ImportError:
                return False

        def _compile_cpp_file(
            self, source_file: Path, output_file: Path
        ) -> tuple[bool, str]:
            """
            Compile a C++ file with TORCH_TARGET_VERSION=2.9.0.
            Returns (success, error_message).
            """
            torch_version_2_9 = "0x0209000000000000"

            cmd = [
                "g++",
                "-c",
                "-std=c++17",
                f"-DTORCH_TARGET_VERSION={torch_version_2_9}",
                f"-I{source_file.parent}",  # For includes in same directory
                *self.pytorch_includes,
            ]

            # Add CUDA flags if available
            if self.cuda_available:
                cmd.extend(self.cuda_includes)

            cmd.extend([str(source_file), "-o", str(output_file)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr

        def _compile_cu_file(
            self, source_file: Path, output_file: Path
        ) -> tuple[bool, str]:
            """
            Compile a CUDA file with TORCH_TARGET_VERSION=2.9.0.
            Returns (success, error_message).
            """
            if not GPU_HOME:
                return False, "one of CUDA_HOME and ROCM_HOME should be set but is not"

            torch_version_2_9 = "0x0209000000000000"

            cmd = [
                os.path.join(GPU_HOME, "bin", "nvcc" if CUDA_HOME else "hipcc"),
                "-c",
                "-std=c++17",
                f"-DTORCH_TARGET_VERSION={torch_version_2_9}",
                f"-I{source_file.parent}",  # For includes in same directory
                *self.pytorch_includes,
                *self.cuda_includes,
            ]

            if ROCM_HOME:
                cmd.extend(["-DUSE_ROCM=1"])

            cmd.extend([str(source_file), "-o", str(output_file)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr

        def _test_function_file(self, source_file: Path):
            """Test that a function file fails to compile with TORCH_TARGET_VERSION=2.9.0."""
            func_name = source_file.stem
            obj_file = self.build_dir / f"{func_name}.o"

            # Choose the appropriate compiler based on file extension
            if source_file.suffix == ".cu":
                if not self.cuda_available:
                    self.skipTest(f"CUDA not available, skipping {source_file.name}")
                success, error_msg = self._compile_cu_file(source_file, obj_file)
            else:
                success, error_msg = self._compile_cpp_file(source_file, obj_file)

            obj_file.unlink(missing_ok=True)

            # Print error details for debugging
            if not success:
                relevant_errors = self._extract_relevant_errors(error_msg)
                if relevant_errors:
                    print(f"\n  Compilation errors for {func_name} (requires 2.10+):")
                    for err in relevant_errors:
                        print(f"    {err}")

            self.assertFalse(
                success,
                f"Function {func_name} compiled successfully with TORCH_TARGET_VERSION=2.9.0. "
                f"This could mean two things.\n\t1. It should run with 2.9.0 and should be "
                "moved to libtorch_agn_2_9_extension\n\t2. The function(s) it tests do not use the "
                "proper TORCH_FEATURE_VERSION guards\n\nThe libtorch_agnostic_targetting CI workflow will "
                "verify if you incorrectly move this to the 2_9 extension instead of adding "
                "the appropriate version guards.",
            )

        def test_mv_tensor_accessor_cpu_works_with_2_9(self):
            """Test that mv_tensor_accessor_cpu.cpp compiles successfully with 2.9.0.

            This is a negative test - it ensures that a file we expect to work with 2.9.0
            actually does compile. This validates that our test infrastructure correctly
            distinguishes between files that require 2.10+ and those that don't.
            """
            cpp_file = self.csrc_dir / "mv_tensor_accessor_cpu.cpp"

            if not cpp_file.exists():
                self.skipTest(f"{cpp_file} not found - this is a test file only")

            obj_file = self.build_dir / "mv_tensor_accessor_cpu.o"
            success, error_msg = self._compile_cpp_file(cpp_file, obj_file)

            # Clean up
            obj_file.unlink(missing_ok=True)

            if not success:
                relevant_errors = self._extract_relevant_errors(error_msg)
                if relevant_errors:
                    print(
                        "\n  Unexpected compilation errors for mv_tensor_accessor_cpu:"
                    )
                    for err in relevant_errors:
                        print(f"{err}")

            self.assertTrue(
                success,
                f"mv_tensor_accessor_cpu.cpp failed to compile with TORCH_TARGET_VERSION=2.9.0. "
                f"This file is expected to work with 2.9.0 since it doesn't use 2.10+ features. "
                f"Error: {error_msg}",
            )

        def test_mv_tensor_accessor_cuda_works_with_2_9(self):
            """Test that mv_tensor_accessor_cuda.cu compiles successfully with 2.9.0.

            This is a negative test - it ensures that a .cu file we expect to work with 2.9.0
            actually does compile. This validates that our test infrastructure correctly
            compiles CUDA files and distinguishes between files that require 2.10+ and those
            that don't.
            """
            if not self.cuda_available:
                self.skipTest(
                    "CUDA not available, skipping mv_tensor_accessor_cuda.cu test"
                )

            cu_file = self.csrc_dir / "mv_tensor_accessor_cuda.cu"

            if not cu_file.exists():
                self.skipTest(f"{cu_file} not found - this is a test file only")

            obj_file = self.build_dir / "cuda_kernel.o"
            success, error_msg = self._compile_cu_file(cu_file, obj_file)

            # Clean up
            obj_file.unlink(missing_ok=True)

            if not success:
                relevant_errors = self._extract_relevant_errors(error_msg)
                if relevant_errors:
                    print(
                        "\n  Unexpected compilation errors for mv_tensor_accessor_cuda.cu:"
                    )
                    for err in relevant_errors:
                        print(f"{err}")

            self.assertTrue(
                success,
                f"mv_tensor_accessor_cuda.cu failed to compile with TORCH_TARGET_VERSION=2.9.0. "
                f"This file is expected to work with 2.9.0 since it doesn't use 2.10+ features. "
                f"Error: {error_msg}",
            )

        @staticmethod
        def _extract_relevant_errors(error_msg: str) -> list[str]:
            """Extract the most relevant error messages."""
            error_lines = error_msg.strip().split("\n")
            relevant_errors = []

            for line in error_lines:
                line_lower = line.lower()
                if (
                    "error:" in line_lower
                    or "undefined" in line_lower
                    or "undeclared" in line_lower
                    or "no member named" in line_lower
                ):
                    relevant_errors.append(line.strip())

            return relevant_errors

    # Dynamically create test methods for each .cpp and .cu file

    def _create_test_method_for_file(source_file: Path):
        """Create a test method for a specific source file."""

        def test_method_impl(self):
            self._test_function_file(source_file)

        # Set a descriptive name and docstring
        func_name = source_file.stem
        file_ext = source_file.suffix
        test_method_impl.__name__ = f"test_{func_name}_requires_2_10"
        test_method_impl.__doc__ = (
            f"Test that {func_name}{file_ext} requires PyTorch 2.10+"
        )

        return test_method_impl

    # Test discovery: generate a test for each .cpp and .cu file
    _csrc_dir = Path(__file__).parent / "csrc"
    assert _csrc_dir.exists()
    # Collect both .cpp and .cu files, excluding those used for negative test
    # already defined above
    _source_files = sorted(
        [f for f in _csrc_dir.rglob("*.cpp") if f.name != "mv_tensor_accessor_cpu.cpp"]
        + [f for f in _csrc_dir.rglob("*.cu") if f.name != "mv_tensor_accessor_cuda.cu"]
    )

    for _source_file in _source_files:
        _test_method = _create_test_method_for_file(_source_file)
        setattr(FunctionVersionCompatibilityTest, _test_method.__name__, _test_method)

    del (
        _create_test_method_for_file,
        _csrc_dir,
        _source_files,
        _source_file,
        _test_method,
    )

if __name__ == "__main__":
    run_tests()
