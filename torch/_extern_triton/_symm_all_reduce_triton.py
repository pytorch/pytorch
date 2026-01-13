# Owner(s): ["oncall: distributed"]
"""
Symmetric all-reduce Triton extern library.

This module provides Triton-compatible extern functions for symmetric memory
all-reduce operations using NCCL's Local Symmetric Access (LSA) API.

Usage:
    from torch._extern_triton import requires_symm_all_reduce, symm_all_reduce_sum_f32

    @requires_symm_all_reduce
    @triton.jit
    def kernel(...):
        result = symm_all_reduce_sum_f32(ctx_ptr, buffer_ptr, offset, num_elements)

Prerequisites:
    1. NCCL >= 2.28.9 with symmetric memory device support
    2. Build the CUDA library to bitcode:
       cd torch/csrc/_extern_triton && make symm_all_reduce.bc CUDA_ARCH=sm_80
    3. Or set SYMM_ALL_REDUCE_LIB_PATH environment variable
"""

from __future__ import annotations

import os
from typing import Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from triton.runtime.jit import JITFunction

try:
    import triton
    from triton import language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


class SymmAllReduceLibFinder:
    """Utility class for finding the symmetric all-reduce bitcode library."""

    _cached_path: Optional[str] = None
    ENV_VAR = "SYMM_ALL_REDUCE_LIB_PATH"

    @classmethod
    def find_device_library(cls) -> str:
        """
        Find the path to the symm_all_reduce.bc library.

        Search order:
        1. SYMM_ALL_REDUCE_LIB_PATH environment variable
        2. torch/lib/symm_all_reduce.bc (installed location)
        3. torch/csrc/_extern_triton/symm_all_reduce.bc (source location)

        Returns:
            str: Absolute path to the .bc file

        Raises:
            RuntimeError: If the library cannot be found
        """
        if cls._cached_path is not None:
            return cls._cached_path

        # Check environment variable first
        env_path = os.environ.get(cls.ENV_VAR)
        if env_path and os.path.isfile(env_path):
            cls._cached_path = os.path.abspath(env_path)
            return cls._cached_path

        # Try installed location (torch/lib)
        import torch

        lib_dir = os.path.dirname(torch.__file__)

        # Check installed lib directory
        installed_path = os.path.join(lib_dir, "lib", "symm_all_reduce.bc")
        if os.path.isfile(installed_path):
            cls._cached_path = os.path.abspath(installed_path)
            return cls._cached_path

        # Check source directory (for development)
        this_dir = os.path.dirname(__file__)
        source_path = os.path.join(
            os.path.dirname(this_dir), "csrc", "_extern_triton", "symm_all_reduce.bc"
        )
        if os.path.isfile(source_path):
            cls._cached_path = os.path.abspath(source_path)
            return cls._cached_path

        # Library not found
        raise RuntimeError(
            f"Symmetric all-reduce bitcode library (symm_all_reduce.bc) not found.\n"
            f"Searched locations:\n"
            f"  - {cls.ENV_VAR} environment variable\n"
            f"  - {installed_path}\n"
            f"  - {source_path}\n\n"
            f"To build the library:\n"
            f"  cd torch/csrc/_extern_triton && make symm_all_reduce.bc CUDA_ARCH=sm_80\n"
            f"Or set the {cls.ENV_VAR} environment variable to point to the .bc file."
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if the library is available."""
        try:
            cls.find_device_library()
            return True
        except RuntimeError:
            return False

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached library path."""
        cls._cached_path = None


def requires_symm_all_reduce_lib():
    """
    DEPRECATED: Use @requires_symm_all_reduce decorator instead.

    This function was a placeholder and has been replaced by the proper
    decorator that handles extern_libs passing.
    """
    import warnings

    warnings.warn(
        "requires_symm_all_reduce_lib() is deprecated. "
        "Use @requires_symm_all_reduce decorator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(func):
        return func

    return decorator


# Triton extern function declarations
if TRITON_AVAILABLE:
    from triton.language import core
    from triton.runtime.jit import JITFunction, KernelInterface

    # Pre-compute the library path at module load time.
    # Triton's JIT compiler parses the source code and hashes any referenced
    # functions, so we must use a plain string constant, not a function call.
    try:
        _SYMM_ALL_REDUCE_LIB_PATH: str = SymmAllReduceLibFinder.find_device_library()
    except RuntimeError:
        # Library not found - set to empty string, will fail at runtime
        _SYMM_ALL_REDUCE_LIB_PATH = ""

    # Create a new Callable class that follows the KernelInterface protocol so
    # that the Callable works with the subscript operator, e.g. `foo[(1, 1)]`
    class GridCallableWithExtern(KernelInterface):
        """
        `KernelInterface` invokes `self.run` in `__getitem__`, i.e. [].  We
        implement a `run` method by directing the call to `JITFunction.run`,
        with added extern_libs kwarg, so that users don't have to pass it.
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args: Any, **kwargs: Any) -> Any:
            # Call the JITFunction.run with added extern_libs kwarg
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def requires_symm_all_reduce(jit_func: JITFunction) -> GridCallableWithExtern:
        """
        A decorator to register a Triton kernel function that requires
        symmetric all-reduce extern library.

        This decorator wraps the JITFunction to automatically pass the
        extern_libs parameter when the kernel is launched.

        Example usage:
        ```
            @requires_symm_all_reduce
            @triton.jit
            def kernel(...):
                result = symm_all_reduce_sum_f32(ctx_ptr, buffer_ptr, offset, n)
        ```

        If you would like to specify a path to the symmetric all-reduce library
        other than standard search locations, you can use the following
        environment variable:
        ```
            export SYMM_ALL_REDUCE_LIB_PATH=/path/to/symm_all_reduce.bc
        ```

        Args:
            jit_func: A JITFunction created by @triton.jit

        Returns:
            GridCallableWithExtern: A wrapper that handles extern_libs passing
        """
        if not isinstance(jit_func, JITFunction):
            raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

        # Find the symmetric all-reduce device library
        lib_path = SymmAllReduceLibFinder.find_device_library()
        extern_libs = {"symm_all_reduce_sum_f32": lib_path}

        return GridCallableWithExtern(jit_func, extern_libs)

    @core.extern
    def symm_all_reduce_sum_f32(
        ctx_ptr,
        local_ptr,
        byte_offset,
        num_elements,
        _semantic=None,
    ):
        """
        Perform all-reduce sum operation on symmetric memory buffers (float32).

        This is a collective operation that must be called by all ranks.
        The context, offset, and num_elements should be the same across all ranks.

        Args:
            ctx_ptr: Pointer to SymmContext (device pointer as int64)
            local_ptr: Pointer to local buffer (device pointer as int64)
            byte_offset: Byte offset within symmetric buffer
            num_elements: Number of float32 elements to reduce

        Returns:
            int32: 0 on success, non-zero on error
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr, byte_offset, num_elements],
            {
                # Support various type combinations that might be passed
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int64"),  # byte_offset
                    core.dtype("int64"),  # num_elements
                ): ("symm_all_reduce_sum_f32", core.dtype("int32")),
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int64"),  # byte_offset
                    core.dtype("int32"),  # num_elements
                ): ("symm_all_reduce_sum_f32", core.dtype("int32")),
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int32"),  # byte_offset
                    core.dtype("int64"),  # num_elements
                ): ("symm_all_reduce_sum_f32", core.dtype("int32")),
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int32"),  # byte_offset
                    core.dtype("int32"),  # num_elements
                ): ("symm_all_reduce_sum_f32", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

else:
    # Triton not available - provide stubs

    def requires_symm_all_reduce(jit_func):  # type: ignore[misc]
        """Stub for when Triton is not available."""
        raise ImportError("Triton is required for requires_symm_all_reduce decorator")

    def symm_all_reduce_sum_f32(*args, **kwargs):
        raise ImportError("Triton is required for symm_all_reduce_sum_f32")


__all__ = [
    "SymmAllReduceLibFinder",
    "requires_symm_all_reduce",
    "requires_symm_all_reduce_lib",  # deprecated
    "symm_all_reduce_sum_f32",
]
