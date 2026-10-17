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
import shutil
import subprocess
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_WINDOWS,
    run_tests,
    TestCase,
)
from torch.utils.cpp_extension import (
    CUDA_HOME,
    get_cxx_compiler,
    include_paths as torch_include_paths,
    ROCM_HOME,
)


TORCH_TARGET_VERSION_2_9 = "0x0209000000000000"

# .cu sources are compiled with one of these toolkits, so the CUDA checks only
# apply when one of them is set up.
GPU_HOME = CUDA_HOME or ROCM_HOME


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


def _compile_cpp_file(source_file: Path, output_file: Path) -> tuple[bool, str]:
    """
    Compile a C++ file with TORCH_TARGET_VERSION=2.9.0.
    Returns (success, error_message).
    """
    cmd = [
        get_cxx_compiler(),
        "-c",
        "-std=c++20",
        f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION_2_9}",
        f"-I{source_file.parent}",  # For includes in same directory
        *[f"-I{path}" for path in torch_include_paths(device_type="cpu")],
        str(source_file),
        "-o",
        str(output_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        return True, ""
    else:
        return False, result.stderr


def _compile_cu_file(source_file: Path, output_file: Path) -> tuple[bool, str]:
    """
    Compile a CUDA file with TORCH_TARGET_VERSION=2.9.0.
    Returns (success, error_message).
    """
    if not GPU_HOME:
        return False, "one of CUDA_HOME and ROCM_HOME should be set but is not"

    gpu_include_path = os.path.join(GPU_HOME, "include")
    gpu_includes = [f"-I{gpu_include_path}"] if os.path.exists(gpu_include_path) else []

    cmd = [
        os.path.join(GPU_HOME, "bin", "nvcc" if CUDA_HOME else "hipcc"),
        "-c",
        "-std=c++20",
        f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION_2_9}",
        f"-I{source_file.parent}",  # For includes in same directory
        *[f"-I{path}" for path in torch_include_paths(device_type="cpu")],
        *gpu_includes,
    ]

    cmd.extend(["-DUSE_CUDA"])
    if ROCM_HOME:
        cmd.extend(["-DUSE_ROCM=1"])

    cmd.extend([str(source_file), "-o", str(output_file)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        return True, ""
    else:
        return False, result.stderr


if not IS_WINDOWS:

    class _FunctionVersionCompatibilityBase(TestCase):
        """Shared build harness for the version compatibility checks."""

        csrc_dir: Path
        csrc_dir_2_9: Path
        build_dir: Path

        @classmethod
        def setUpClass(cls):
            """Set up test environment once for all tests."""
            ext_dir = Path(__file__).parent
            cls.csrc_dir = ext_dir / "csrc"
            cls.csrc_dir_2_9 = ext_dir.parent / "libtorch_agn_2_9_extension" / "csrc"
            cls.build_dir = Path(tempfile.mkdtemp(prefix="version_check_"))

        @classmethod
        def tearDownClass(cls):
            """Clean up build directory."""
            if cls.build_dir.exists():
                shutil.rmtree(cls.build_dir)

        def _assert_requires_2_10(
            self, func_name: str, success: bool, error_msg: str
        ) -> None:
            """Assert that a source requiring 2.10+ failed to build with 2.9.0."""
            if not success:
                relevant_errors = _extract_relevant_errors(error_msg)
                if relevant_errors:
                    print(f"\n  Compilation errors for {func_name} (requires 2.10+):")
                    for err in relevant_errors:
                        print(f"    {err}")

            self.assertFalse(
                success,
                lambda msg: f"{msg}\nFunction {func_name} compiled successfully with TORCH_TARGET_VERSION=2.9.0. "
                f"This could mean two things.\n\t1. It should run with 2.9.0 and should be "
                "moved to libtorch_agn_2_9_extension\n\t2. The function(s) it tests do not use the "
                "proper TORCH_FEATURE_VERSION guards\n\nThe libtorch_agnostic_targetting CI workflow will "
                "verify if you incorrectly move this to the 2_9 extension instead of adding "
                "the appropriate version guards.",
            )

    class FunctionVersionCompatibilityTestGeneric(_FunctionVersionCompatibilityBase):
        """Test that all C++ function files require PyTorch 2.10+.

        The .cpp sources are built with the host compiler against the CPU
        headers, so these tests do not need an accelerator.
        """

        hw_classification = HardwareClassification.GENERIC

        def _test_function_file(self, source_file: Path):
            """Test that a function file fails to compile with TORCH_TARGET_VERSION=2.9.0."""
            func_name = source_file.stem
            obj_file = self.build_dir / f"{func_name}.o"

            success, error_msg = _compile_cpp_file(source_file, obj_file)

            obj_file.unlink(missing_ok=True)

            self._assert_requires_2_10(func_name, success, error_msg)

        def test_kernel_works_with_2_9(self):
            """Test that the 2.9 extension's kernel.cpp compiles successfully with 2.9.0.

            This is a control test - it ensures that a file we expect to work with 2.9.0
            actually does compile. Every other test in this suite asserts that
            compilation *fails* for sources that require 2.10+, so we have this test to
            validate that our test infra correctly distinguishes when a source can compile
            fine with 2.9.
            """
            cpp_file = self.csrc_dir_2_9 / "kernel.cpp"
            self.assertTrue(cpp_file.exists(), f"{cpp_file} does not exist")

            obj_file = self.build_dir / "kernel.o"
            success, error_msg = _compile_cpp_file(cpp_file, obj_file)

            # Clean up
            obj_file.unlink(missing_ok=True)

            if not success:
                relevant_errors = _extract_relevant_errors(error_msg)
                if relevant_errors:
                    print("\n  Unexpected compilation errors for kernel.cpp:")
                    for err in relevant_errors:
                        print(f"{err}")

            self.assertTrue(
                success,
                lambda msg: f"{msg}\nkernel.cpp failed to compile with TORCH_TARGET_VERSION=2.9.0. "
                f"This file is expected to work with 2.9.0 since it doesn't use 2.10+ features. "
                f"Error: {error_msg}",
            )

    class FunctionVersionCompatibilityTestCUDA(_FunctionVersionCompatibilityBase):
        """Test that the CUDA function files require PyTorch 2.10+.

        The .cu sources are built with nvcc or hipcc, so these tests are only
        run when one of those toolkits is available.
        """

        hw_classification = HardwareClassification.CUDA

        def _test_function_file(self, source_file: Path):
            """Test that a function file fails to compile with TORCH_TARGET_VERSION=2.9.0."""
            if not GPU_HOME:
                self.skipTest(
                    f"neither CUDA_HOME nor ROCM_HOME is set, skipping {source_file.name}"
                )

            func_name = source_file.stem
            obj_file = self.build_dir / f"{func_name}.o"

            success, error_msg = _compile_cu_file(source_file, obj_file)

            obj_file.unlink(missing_ok=True)

            self._assert_requires_2_10(func_name, success, error_msg)

        def test_cuda_kernel_works_with_2_9(self):
            """Test that cuda_kernel.cu compiles successfully with 2.9.0.

            This is a control test - it ensures that a .cu file we expect to work with 2.9.0
            actually does compile. This validates that our test infrastructure correctly
            compiles CUDA files and distinguishes between files that require 2.10+ and those
            that don't.
            """
            if not GPU_HOME:
                self.skipTest(
                    "neither CUDA_HOME nor ROCM_HOME is set, skipping cuda_kernel.cu test"
                )

            cu_file = self.csrc_dir_2_9 / "cuda_kernel.cu"
            self.assertTrue(cu_file.exists(), f"{cu_file} does not exist")

            obj_file = self.build_dir / "cuda_kernel.o"
            success, error_msg = _compile_cu_file(cu_file, obj_file)

            # Clean up
            obj_file.unlink(missing_ok=True)

            if not success:
                relevant_errors = _extract_relevant_errors(error_msg)
                if relevant_errors:
                    print("\n  Unexpected compilation errors for cuda_kernel.cu:")
                    for err in relevant_errors:
                        print(f"{err}")

            self.assertTrue(
                success,
                lambda msg: f"{msg}\ncuda_kernel.cu failed to compile with TORCH_TARGET_VERSION=2.9.0. "
                f"This file is expected to work with 2.9.0 since it doesn't use 2.10+ features. "
                f"Error: {error_msg}",
            )

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
    if not _csrc_dir.exists():
        raise AssertionError(f"Expected csrc directory to exist at {_csrc_dir}")
    # Collect both .cpp and .cu files. The control tests defined above compile
    # sources from the 2.9 extension, which is not globbed here.
    _source_files = sorted([*_csrc_dir.rglob("*.cpp"), *_csrc_dir.rglob("*.cu")])

    for _source_file in _source_files:
        _test_method = _create_test_method_for_file(_source_file)
        # .cu sources need a CUDA or ROCm toolkit, the .cpp sources do not.
        _test_class = (
            FunctionVersionCompatibilityTestCUDA
            if _source_file.suffix == ".cu"
            else FunctionVersionCompatibilityTestGeneric
        )
        setattr(_test_class, _test_method.__name__, _test_method)

    del (
        _create_test_method_for_file,
        _csrc_dir,
        _source_files,
        _source_file,
        _test_method,
        _test_class,
    )

if __name__ == "__main__":
    run_tests()
