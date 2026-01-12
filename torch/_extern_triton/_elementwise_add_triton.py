# Owner(s): ["oncall: distributed"]
"""
Triton wrapper for elementwise addition CUDA kernels.

This module provides Triton-compatible wrappers for elementwise addition
operations implemented in CUDA. The CUDA kernels are compiled to LLVM
bitcode and linked with Triton kernels via the extern_libs mechanism.

Usage:
    from torch._extern_triton import requires_elementwise_add_lib, scalar_add_f32

    @requires_elementwise_add_lib
    @triton.jit
    def my_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)

        # Use the external CUDA kernel for addition
        result = scalar_add_f32(a, b)

        tl.store(out_ptr + offsets, result, mask=mask)

To compile the CUDA kernel to bitcode:
    cd torch/csrc/_extern_triton
    make CUDA_ARCH=sm_80

Environment Variables:
    ELEMENTWISE_ADD_LIB_PATH: Optional path to the compiled .bc file
"""

import logging
import os
from typing import Any

from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class ElementwiseAddLibFinder:
    """
    A class to find the path to the elementwise add device library.

    Environment variable:

    `ELEMENTWISE_ADD_LIB_PATH` (Optional[str]): The path where the elementwise
    add device library is located. If not provided, it will search for the
    library relative to the torch installation.
    """

    # Class variable to store the found library path for reuse
    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        """
        Find the path to the elementwise add device library.

        Returns:
            str: The path to elementwise_add.bc.
        """
        if cls.found_device_lib_path is not None:
            # Return the cached path if it exists
            return cls.found_device_lib_path

        # First, check if the user has specified a custom library path
        user_lib_path = os.environ.get("ELEMENTWISE_ADD_LIB_PATH", None)
        if user_lib_path is not None:
            if not os.path.exists(user_lib_path):
                raise RuntimeError(
                    f"Elementwise add library not found at specified path: {user_lib_path}"
                )
            cls.found_device_lib_path = user_lib_path
            return user_lib_path

        # Otherwise, search for the library in default paths
        paths = []

        # Check relative to this file (in PyTorch source tree)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_path = os.path.join(
            os.path.dirname(this_dir), "csrc", "_extern_triton", "elementwise_add.bc"
        )
        paths.append(csrc_path)

        # Check in torch lib directory (installed package)
        try:
            import torch

            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            lib_path = os.path.join(torch_lib, "elementwise_add.bc")
            paths.append(lib_path)

            # Also check CMAKE_LIBRARY_OUTPUT_DIRECTORY path (build directory)
            build_lib = os.path.join(
                os.path.dirname(os.path.dirname(torch.__file__)),
                "build",
                "lib",
                "elementwise_add.bc",
            )
            paths.append(build_lib)
        except ImportError:
            pass

        # Common system paths
        common_paths = [
            "/usr/local/lib/elementwise_add.bc",
            "/usr/lib/elementwise_add.bc",
        ]
        paths.extend(common_paths)

        for path in paths:
            if os.path.exists(path):
                cls.found_device_lib_path = path
                return path

        raise RuntimeError(
            f"Elementwise add library not found. Searched: {paths}\n"
            "Please compile the CUDA source to bitcode using:\n"
            "  cd torch/csrc/_extern_triton && make CUDA_ARCH=sm_80\n"
            "Or set ELEMENTWISE_ADD_LIB_PATH environment variable."
        )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    import triton
    import triton.language as tl
    from triton.language import core

    # =========================================================================
    # EXTERN LIBRARY WRAPPER CLASS
    # Follows the pattern from nvshmem_triton.py
    # =========================================================================

    class GridCallableWithExtern(KernelInterface):
        """
        `KernelInterface` invokes `self.run` in `__getitem__`, i.e. [].  We
        implement a `run` method by directing the call to `JITFunction.run`,
        with added extern_libs kwarg, so that users don't have to pass it
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args: Any, **kwargs: Any) -> Any:
            # Call the JITFunction.run with added extern_libs kwarg
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def requires_elementwise_add_lib(
        jit_func: JITFunction,
    ) -> GridCallableWithExtern:
        """
        A decorator to register a Triton kernel function that requires the
        elementwise add external library.

        Example usage:
            @requires_elementwise_add_lib
            @triton.jit
            def my_add_kernel(...):
                result = scalar_add_f32(a, b)
                ...

        If you would like to specify a custom path to the library, you can use
        the following environment variable:
            export ELEMENTWISE_ADD_LIB_PATH=/path/to/elementwise_add.bc
        """
        if not isinstance(jit_func, JITFunction):
            raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

        # Find the elementwise add device library
        lib_path = ElementwiseAddLibFinder.find_device_library()
        extern_libs = {"elementwise_add": lib_path}

        return GridCallableWithExtern(jit_func, extern_libs)

    # =========================================================================
    # LOW-LEVEL EXTERN ELEMENTWISE WRAPPERS
    # These define the mapping from Triton types to CUDA function names
    # =========================================================================

    @core.extern
    def _scalar_add_f32_extern(a, b, _semantic=None):  # type: ignore[no-untyped-def]
        """
        Low-level extern wrapper for scalar float32 addition.

        Maps to: float scalar_add_f32(float a, float b)
        """
        return core.extern_elementwise(
            "",  # No inline PTX assembly needed - using external library
            "",  # No constraints needed
            [a, b],
            {
                # Input types -> (function_name, return_type)
                (core.dtype("fp32"), core.dtype("fp32")): (
                    "scalar_add_f32",
                    core.dtype("fp32"),
                ),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def _scalar_add_f16_extern(a, b, _semantic=None):  # type: ignore[no-untyped-def]
        """
        Low-level extern wrapper for scalar float16 addition.

        Maps to: half scalar_add_f16(half a, half b)
        """
        return core.extern_elementwise(
            "",
            "",
            [a, b],
            {
                (core.dtype("fp16"), core.dtype("fp16")): (
                    "scalar_add_f16",
                    core.dtype("fp16"),
                ),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def _scalar_add_f64_extern(a, b, _semantic=None):  # type: ignore[no-untyped-def]
        """
        Low-level extern wrapper for scalar float64 addition.

        Maps to: double scalar_add_f64(double a, double b)
        """
        return core.extern_elementwise(
            "",
            "",
            [a, b],
            {
                (core.dtype("fp64"), core.dtype("fp64")): (
                    "scalar_add_f64",
                    core.dtype("fp64"),
                ),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    # =========================================================================
    # HIGH-LEVEL TRITON JIT WRAPPERS
    # These are the user-facing functions that can be called from Triton kernels
    # =========================================================================

    @triton.jit
    def scalar_add_f32(a: tl.tensor, b: tl.tensor) -> tl.tensor:
        """
        Scalar element-wise addition for float32.

        This function performs element-wise addition using an external CUDA
        kernel. Each element is processed independently using the scalar_add_f32
        function from the external library.

        Args:
            a: First operand (block of fp32 values)
            b: Second operand (block of fp32 values, same shape as a)

        Returns:
            Element-wise sum a + b

        Note:
            The kernel containing this function must be decorated with
            @requires_elementwise_add_lib to link the external library.
        """
        return _scalar_add_f32_extern(a, b)

    @triton.jit
    def scalar_add_f16(a: tl.tensor, b: tl.tensor) -> tl.tensor:
        """
        Scalar element-wise addition for float16.

        This function performs element-wise addition using an external CUDA
        kernel with native half-precision arithmetic.

        Args:
            a: First operand (block of fp16 values)
            b: Second operand (block of fp16 values, same shape as a)

        Returns:
            Element-wise sum a + b

        Note:
            The kernel containing this function must be decorated with
            @requires_elementwise_add_lib to link the external library.
        """
        return _scalar_add_f16_extern(a, b)

    @triton.jit
    def scalar_add_f64(a: tl.tensor, b: tl.tensor) -> tl.tensor:
        """
        Scalar element-wise addition for float64.

        This function performs element-wise addition using an external CUDA
        kernel with double precision.

        Args:
            a: First operand (block of fp64 values)
            b: Second operand (block of fp64 values, same shape as a)

        Returns:
            Element-wise sum a + b

        Note:
            The kernel containing this function must be decorated with
            @requires_elementwise_add_lib to link the external library.
        """
        return _scalar_add_f64_extern(a, b)

else:
    # Triton not available - provide stub implementations that raise errors

    def requires_elementwise_add_lib(jit_func: Any) -> Any:
        """Stub that raises error when Triton is not available."""
        raise RuntimeError(
            "requires_elementwise_add_lib requires Triton to be installed."
        )

    def scalar_add_f32(a: Any, b: Any) -> Any:
        """Stub that raises error when Triton is not available."""
        raise RuntimeError("scalar_add_f32 requires Triton to be installed.")

    def scalar_add_f16(a: Any, b: Any) -> Any:
        """Stub that raises error when Triton is not available."""
        raise RuntimeError("scalar_add_f16 requires Triton to be installed.")

    def scalar_add_f64(a: Any, b: Any) -> Any:
        """Stub that raises error when Triton is not available."""
        raise RuntimeError("scalar_add_f64 requires Triton to be installed.")
