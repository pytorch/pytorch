# Owner(s): ["oncall: distributed"]
"""
Unified symmetric all-reduce Triton extern library.

This module provides Triton-compatible extern functions for symmetric memory
all-reduce operations with automatic backend dispatch (NCCL or NVSHMEM).

The unified frontend function `symm_all_reduce_sum_f32` dispatches to the
appropriate backend based on the SymmContext type passed as the first argument.

Usage:
    from torch._extern_triton import (
        requires_symm_all_reduce,
        symm_all_reduce_sum_f32,
    )

    @requires_symm_all_reduce
    @triton.jit
    def kernel(...):
        result = symm_all_reduce_sum_f32(ctx_ptr, buffer_ptr, offset, num_elements)

Prerequisites:
    - For NVSHMEM: libnvshmem_device.bc (fully functional)
    - For NCCL: libnccl_device.bc (not available - NCCL does not ship this)

Note: Currently only NVSHMEM backend is functional because NCCL does not
provide a device bitcode library. The NCCL path will compile but fail at
PTX generation time with unresolved symbols.
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
    """Utility class for finding the unified symmetric all-reduce bitcode library."""

    _cached_bc_path: Optional[str] = None
    _cached_nvshmem_path: Optional[str] = None
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
        if cls._cached_bc_path is not None:
            return cls._cached_bc_path

        # Check environment variable first
        env_path = os.environ.get(cls.ENV_VAR)
        if env_path and os.path.isfile(env_path):
            cls._cached_bc_path = os.path.abspath(env_path)
            return cls._cached_bc_path

        # Try installed location (torch/lib)
        import torch

        lib_dir = os.path.dirname(torch.__file__)

        # Check installed lib directory
        installed_path = os.path.join(lib_dir, "lib", "symm_all_reduce.bc")
        if os.path.isfile(installed_path):
            cls._cached_bc_path = os.path.abspath(installed_path)
            return cls._cached_bc_path

        # Check source directory (for development)
        this_dir = os.path.dirname(__file__)
        source_path = os.path.join(
            os.path.dirname(this_dir), "csrc", "_extern_triton", "symm_all_reduce.bc"
        )
        if os.path.isfile(source_path):
            cls._cached_bc_path = os.path.abspath(source_path)
            return cls._cached_bc_path

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
    def find_nvshmem_device_library(cls) -> str:
        """
        Find the path to the libnvshmem_device.bc library.

        This is needed because NVSHMEM functions are called from symm_all_reduce.bc
        and must be linked together.

        Returns:
            str: Absolute path to libnvshmem_device.bc

        Raises:
            RuntimeError: If the library cannot be found
        """
        if cls._cached_nvshmem_path is not None:
            return cls._cached_nvshmem_path

        try:
            from torch.distributed._symmetric_memory._nvshmem_triton import (
                NvshmemLibFinder,
            )

            cls._cached_nvshmem_path = NvshmemLibFinder.find_device_library()
            return cls._cached_nvshmem_path
        except (ImportError, RuntimeError) as e:
            raise RuntimeError(
                f"NVSHMEM device library not found: {e}\n"
                "Make sure NVSHMEM is installed and libnvshmem_device.bc is available.\n"
                "You can set NVSHMEM_LIB_DIR environment variable to specify the location."
            ) from e

    @classmethod
    def is_available(cls) -> bool:
        """Check if the library is available."""
        try:
            cls.find_device_library()
            return True
        except RuntimeError:
            return False

    @classmethod
    def is_nvshmem_available(cls) -> bool:
        """Check if NVSHMEM device library is available."""
        try:
            cls.find_nvshmem_device_library()
            return True
        except RuntimeError:
            return False

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached library paths."""
        cls._cached_bc_path = None
        cls._cached_nvshmem_path = None


# Triton extern function declarations
if TRITON_AVAILABLE:
    from triton.language import core
    from triton.runtime.jit import JITFunction, KernelInterface

    # Pre-compute library paths at module load time
    try:
        _SYMM_ALL_REDUCE_LIB_PATH: str = SymmAllReduceLibFinder.find_device_library()
    except RuntimeError:
        _SYMM_ALL_REDUCE_LIB_PATH = ""

    try:
        _NVSHMEM_LIB_PATH: str = SymmAllReduceLibFinder.find_nvshmem_device_library()
    except RuntimeError:
        _NVSHMEM_LIB_PATH = ""

    # Registry to track kernels that need NVSHMEM initialization
    class SymmAllReduceKernelRegistry:
        """Registry for kernels that require NVSHMEM CUModule initialization."""

        _to_init: dict[str, Any] = {}

        @classmethod
        def register(cls, name: str) -> None:
            cls._to_init.setdefault(name)

        @classmethod
        def has(cls, name: str) -> bool:
            return name in cls._to_init

    def _symm_all_reduce_init_hook(*args: Any, **kwargs: Any) -> None:
        """Post-compile hook to initialize NVSHMEM CUModule."""
        try:
            from torch._C._distributed_c10d import _nvshmemx_cumodule_init
        except ImportError:
            return

        jit_function = kwargs["fn"].jit_function
        fn_name = jit_function.fn.__name__

        if SymmAllReduceKernelRegistry.has(fn_name):
            key = kwargs["key"]
            device = kwargs["compile"]["device"]
            kernel_cache = jit_function.device_caches[device][0]
            kernel = kernel_cache.get(key, None)
            if kernel is not None:
                kernel.run  # Ensure kernel is compiled
                _nvshmemx_cumodule_init(kernel.module)

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
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def requires_symm_all_reduce(
        jit_func: JITFunction,
    ) -> GridCallableWithExtern:
        """
        A decorator to register a Triton kernel function that requires
        unified symmetric all-reduce extern library.

        This decorator wraps the JITFunction to automatically pass the
        extern_libs parameter when the kernel is launched. It links both
        symm_all_reduce.bc and libnvshmem_device.bc (for NVSHMEM support).

        It also registers a post-compile hook to initialize the NVSHMEM
        device state in the CUmodule, which is required for NVSHMEM device
        functions (nvshmem_ptr, nvshmemx_barrier_all_block, etc.) to work.

        Example usage:
        ```
            @requires_symm_all_reduce
            @triton.jit
            def kernel(...):
                result = symm_all_reduce_sum_f32(ctx_ptr, buffer_ptr, offset, n)
        ```

        Environment variables for custom library paths:
        - SYMM_ALL_REDUCE_LIB_PATH: Path to symm_all_reduce.bc
        - NVSHMEM_LIB_DIR: Directory containing libnvshmem_device.bc

        Args:
            jit_func: A JITFunction created by @triton.jit

        Returns:
            GridCallableWithExtern: A wrapper that handles extern_libs passing
        """
        import triton

        if not isinstance(jit_func, JITFunction):
            raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

        extern_libs = {}

        # Add the unified symm_all_reduce library
        try:
            bc_path = SymmAllReduceLibFinder.find_device_library()
            extern_libs["symm_all_reduce"] = bc_path
        except RuntimeError:
            pass  # Library not found, will fail at runtime

        # Add NVSHMEM device library (needed for NVSHMEM backend)
        try:
            nvshmem_path = SymmAllReduceLibFinder.find_nvshmem_device_library()
            extern_libs["libnvshmem_device"] = nvshmem_path
        except RuntimeError:
            pass  # NVSHMEM not available, only NCCL backend will work

        # Register the kernel for NVSHMEM CUModule initialization
        SymmAllReduceKernelRegistry.register(jit_func.fn.__name__)

        # Register the post-compile hook for NVSHMEM initialization
        # This is a global setting; filtering is done in the hook itself
        triton.knobs.runtime.jit_post_compile_hook = _symm_all_reduce_init_hook

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
        Perform unified all-reduce sum operation on symmetric memory buffers (float32).

        This is the frontend function that dispatches to either NCCL or NVSHMEM
        backend based on the SymmContext type (ctx_ptr->type field).

        This is a collective operation that must be called by all ranks/PEs.
        The context, offset, and num_elements should be the same across all ranks.

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            local_ptr: Pointer to local buffer (device pointer as int64)
            byte_offset: Byte offset within symmetric buffer (int32)
            num_elements: Number of float32 elements to reduce (int32)

        Returns:
            int32: 0 on success, negative on error
                   -1: null context
                   -2: unknown context type
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr, byte_offset, num_elements],
            {
                # C function signature: (int64, int64, int32, int32) -> int32
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

    def symm_all_reduce_sum_f32(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_all_reduce_sum_f32")


__all__ = [
    "SymmAllReduceLibFinder",
    "requires_symm_all_reduce",
    "symm_all_reduce_sum_f32",
]
