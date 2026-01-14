# Owner(s): ["oncall: distributed"]
"""
Torch symmetric memory Triton extern library.

This module provides Triton-compatible extern functions for symmetric memory
operations with automatic backend dispatch (NCCL or NVSHMEM).

Backend Hint Support:
The symmetric memory primitives (e.g., `symm_all_reduce`) accept an
optional `backend` constexpr argument that controls dispatch:
- BACKEND_DEFAULT (0): Use runtime dispatch based on SymmContext type
- BACKEND_NCCL (1): Dispatch directly to NCCL backend (requires libnccl_device.bc)
- BACKEND_NVSHMEM (2): Dispatch directly to NVSHMEM backend

When a specific backend is chosen, only that backend's bitcode library is needed,
avoiding the need for all backend dependencies.

Usage Pattern:
    The recommended pattern is to define a single kernel that accepts a backend
    hint as a constexpr parameter. Both the decorator and the kernel receive the
    same backend hint, which is then passed down to the torch_symm primitives.

    ```python
    from torch._extern_triton import (
        BACKEND_DEFAULT,
        BACKEND_NVSHMEM,
        DTYPE_FLOAT32,
        REDUCE_OP_SUM,
        requires_torch_symm,
        symm_all_reduce,
    )


    def make_my_kernel(backend: int):
        '''Factory to create kernel with specific backend hint.'''

        @requires_torch_symm(backend=backend)
        @triton.jit
        def my_kernel(
            ctx_ptr,
            buffer_ptr,
            num_elements: tl.constexpr,
            backend_hint: tl.constexpr,  # Same value as decorator's backend
        ):
            byte_offset: tl.int64 = 0
            n_elems: tl.int64 = num_elements
            # Pass backend_hint to primitives for compile-time dispatch
            result = symm_all_reduce(
                ctx_ptr, buffer_ptr, byte_offset, n_elems,
                REDUCE_OP_SUM, DTYPE_FLOAT32, backend_hint
            )

        return my_kernel


    # Create kernel variants for different backends
    my_kernel_dynamic = make_my_kernel(BACKEND_DEFAULT)  # Runtime dispatch
    my_kernel_nvshmem = make_my_kernel(BACKEND_NVSHMEM)  # Direct NVSHMEM

    # Launch with matching backend hint
    my_kernel_nvshmem[(1,)](ctx_ptr, buf_ptr, n_elems, BACKEND_NVSHMEM)
    ```

    The factory pattern ensures that:
    1. The decorator links the correct bitcode libraries
    2. The kernel receives the backend hint as a constexpr
    3. The primitives dispatch at compile-time (not runtime) when hint != DEFAULT

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


# Backend hint values (matching SymmContext::Type enum + DEFAULT)
# Use raw integers for type checking and documentation
_BACKEND_DEFAULT = 0  # Use runtime dispatch based on context type
_BACKEND_NCCL = 1  # Dispatch directly to NCCL backend
_BACKEND_NVSHMEM = 2  # Dispatch directly to NVSHMEM backend

# For use in non-Triton code (type checking, documentation, etc.)
BACKEND_DEFAULT = _BACKEND_DEFAULT
BACKEND_NCCL = _BACKEND_NCCL
BACKEND_NVSHMEM = _BACKEND_NVSHMEM

# Reduction operation constants (matching CUDA constants)
_REDUCE_OP_SUM = 0

REDUCE_OP_SUM = _REDUCE_OP_SUM

# Data type constants (matching CUDA constants)
_DTYPE_FLOAT32 = 0

DTYPE_FLOAT32 = _DTYPE_FLOAT32


class TorchSymmLibFinder:
    """Utility class for finding the torch symmetric memory bitcode library."""

    _cached_bc_path: Optional[str] = None
    _cached_nvshmem_path: Optional[str] = None
    ENV_VAR = "TORCH_SYMM_LIB_PATH"

    @classmethod
    def find_device_library(cls) -> str:
        """
        Find the path to the torch_symm.bc library.

        Search order:
        1. TORCH_SYMM_LIB_PATH environment variable
        2. torch/lib/torch_symm.bc (installed location)
        3. torch/csrc/_extern_triton/torch_symm.bc (source location)

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
        installed_path = os.path.join(lib_dir, "lib", "torch_symm.bc")
        if os.path.isfile(installed_path):
            cls._cached_bc_path = os.path.abspath(installed_path)
            return cls._cached_bc_path

        # Check source directory (for development)
        this_dir = os.path.dirname(__file__)
        source_path = os.path.join(
            os.path.dirname(this_dir), "csrc", "_extern_triton", "torch_symm.bc"
        )
        if os.path.isfile(source_path):
            cls._cached_bc_path = os.path.abspath(source_path)
            return cls._cached_bc_path

        # Library not found
        raise RuntimeError(
            f"Torch symmetric memory bitcode library (torch_symm.bc) not found.\n"
            f"Searched locations:\n"
            f"  - {cls.ENV_VAR} environment variable\n"
            f"  - {installed_path}\n"
            f"  - {source_path}\n\n"
            f"To build the library:\n"
            f"  cd torch/csrc/_extern_triton && make torch_symm.bc CUDA_ARCH=sm_80\n"
            f"Or set the {cls.ENV_VAR} environment variable to point to the .bc file."
        )

    @classmethod
    def find_nvshmem_device_library(cls) -> str:
        """
        Find the path to the libnvshmem_device.bc library.

        This is needed because NVSHMEM functions are called from torch_symm.bc
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
        _TORCH_SYMM_LIB_PATH: str = TorchSymmLibFinder.find_device_library()
    except RuntimeError:
        _TORCH_SYMM_LIB_PATH = ""

    try:
        _NVSHMEM_LIB_PATH: str = TorchSymmLibFinder.find_nvshmem_device_library()
    except RuntimeError:
        _NVSHMEM_LIB_PATH = ""

    # Registry to track kernels that need NVSHMEM initialization
    class TorchSymmKernelRegistry:
        """Registry for kernels that require NVSHMEM CUModule initialization."""

        _to_init: dict[str, Any] = {}

        @classmethod
        def register(cls, name: str) -> None:
            cls._to_init.setdefault(name)

        @classmethod
        def has(cls, name: str) -> bool:
            return name in cls._to_init

    def _torch_symm_init_hook(*args: Any, **kwargs: Any) -> None:
        """Post-compile hook to initialize NVSHMEM CUModule."""
        try:
            from torch._C._distributed_c10d import _nvshmemx_cumodule_init
        except ImportError:
            return

        jit_function = kwargs["fn"].jit_function
        fn_name = jit_function.fn.__name__

        if TorchSymmKernelRegistry.has(fn_name):
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

    def requires_torch_symm(
        jit_func_or_backend: JITFunction | int | None = None,
        backend: int = _BACKEND_DEFAULT,
    ) -> GridCallableWithExtern | Any:
        """
        A decorator to register a Triton kernel function that requires
        torch symmetric memory primitives extern library.

        This decorator wraps the JITFunction to automatically pass the
        extern_libs parameter when the kernel is launched. The libraries
        linked depend on the backend hint:

        - BACKEND_DEFAULT (0): Links torch_symm.bc and libnvshmem_device.bc
          for runtime dispatch based on SymmContext type.
        - BACKEND_NVSHMEM (2): Links torch_symm.bc and libnvshmem_device.bc
          for direct NVSHMEM dispatch (only NVSHMEM bitcode strictly needed).

        It also registers a post-compile hook to initialize the NVSHMEM
        device state in the CUmodule, which is required for NVSHMEM device
        functions (nvshmem_ptr, nvshmemx_barrier_all_block, etc.) to work.

        Recommended Usage Pattern:
            Use a factory function to create kernels with a specific backend
            hint. Both the decorator and the kernel should receive the same
            backend hint, which is passed down to the torch_symm primitives.

            ```python
            def make_my_kernel(backend: int):
                @requires_torch_symm(backend=backend)
                @triton.jit
                def my_kernel(
                    ctx_ptr,
                    buffer_ptr,
                    num_elements: tl.constexpr,
                    backend_hint: tl.constexpr,
                ):
                    byte_offset: tl.int64 = 0
                    n_elems: tl.int64 = num_elements
                    result = symm_all_reduce(
                        ctx_ptr, buffer_ptr, byte_offset, n_elems,
                        REDUCE_OP_SUM, DTYPE_FLOAT32, backend_hint
                    )

                return my_kernel


            # Create variants
            my_kernel_dynamic = make_my_kernel(BACKEND_DEFAULT)
            my_kernel_nvshmem = make_my_kernel(BACKEND_NVSHMEM)

            # Launch with matching backend hint
            my_kernel_nvshmem[(1,)](ctx, buf, n, BACKEND_NVSHMEM)
            ```

        Environment variables for custom library paths:
        - TORCH_SYMM_LIB_PATH: Path to torch_symm.bc
        - NVSHMEM_LIB_DIR: Directory containing libnvshmem_device.bc

        Args:
            jit_func_or_backend: Either a JITFunction (when used without args)
                                 or backend hint (when used with args)
            backend: Backend hint (default=BACKEND_DEFAULT)
                     - BACKEND_DEFAULT (0): Runtime dispatch
                     - BACKEND_NVSHMEM (2): Direct NVSHMEM dispatch

        Returns:
            GridCallableWithExtern: A wrapper that handles extern_libs passing
        """
        import triton

        def _apply_decorator(
            jit_func: JITFunction, backend_hint: int
        ) -> GridCallableWithExtern:
            if not isinstance(jit_func, JITFunction):
                raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

            extern_libs = {}

            # Add the unified torch_symm library
            try:
                bc_path = TorchSymmLibFinder.find_device_library()
                extern_libs["torch_symm"] = bc_path
            except RuntimeError:
                pass  # Library not found, will fail at runtime

            # Add NVSHMEM device library based on backend hint
            # NVSHMEM is required for both DEFAULT and NVSHMEM backends
            # (NCCL has no device bitcode library)
            if backend_hint in (_BACKEND_DEFAULT, _BACKEND_NVSHMEM):
                try:
                    nvshmem_path = TorchSymmLibFinder.find_nvshmem_device_library()
                    extern_libs["libnvshmem_device"] = nvshmem_path
                except RuntimeError as e:
                    backend_name = (
                        "BACKEND_DEFAULT"
                        if backend_hint == _BACKEND_DEFAULT
                        else "BACKEND_NVSHMEM"
                    )
                    raise RuntimeError(
                        f"NVSHMEM device library required for {backend_name}: {e}"
                    ) from e

            # Register the kernel for NVSHMEM CUModule initialization
            TorchSymmKernelRegistry.register(jit_func.fn.__name__)

            # Register the post-compile hook for NVSHMEM initialization
            # This is a global setting; filtering is done in the hook itself
            triton.knobs.runtime.jit_post_compile_hook = _torch_symm_init_hook

            return GridCallableWithExtern(jit_func, extern_libs)

        # Handle different calling conventions:
        # 1. @requires_torch_symm (no parentheses) - jit_func_or_backend is JITFunction
        # 2. @requires_torch_symm() - jit_func_or_backend is None
        # 3. @requires_torch_symm(backend=X) - jit_func_or_backend is None, backend is X
        # 4. @requires_torch_symm(BACKEND_NVSHMEM) - jit_func_or_backend is int

        if jit_func_or_backend is None:
            # Called with parentheses but no positional arg: @requires_torch_symm()
            # or @requires_torch_symm(backend=X)
            def decorator(jit_func: JITFunction) -> GridCallableWithExtern:
                return _apply_decorator(jit_func, backend)

            return decorator
        elif isinstance(jit_func_or_backend, int):
            # Called with backend as positional arg: @requires_torch_symm(BACKEND_NVSHMEM)
            backend_hint = jit_func_or_backend

            def decorator(jit_func: JITFunction) -> GridCallableWithExtern:
                return _apply_decorator(jit_func, backend_hint)

            return decorator
        else:
            # Called without parentheses: @requires_torch_symm
            return _apply_decorator(jit_func_or_backend, _BACKEND_DEFAULT)

    @core.extern
    def _symm_all_reduce_frontend(
        ctx_ptr,
        local_ptr,
        byte_offset,
        num_elements,
        reduce_op,
        dtype,
        _semantic=None,
    ):
        """
        Frontend all-reduce operation that dispatches based on SymmContext type.

        DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
        NOT efficient. It is provided solely to demonstrate the symmetric memory
        abstraction layer API. This implementation should NOT be used as a reference
        for production kernels and is NOT part of the proposed set of kernels that
        constitute the symmetric memory abstraction layer.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr, byte_offset, num_elements, reduce_op, dtype],
            {
                # C function signature: (int64, int64, int32, int32, int32, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int32"),  # byte_offset
                    core.dtype("int32"),  # num_elements
                    core.dtype("int32"),  # reduce_op
                    core.dtype("int32"),  # dtype
                ): ("symm_all_reduce", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_all_reduce(
        ctx_ptr,
        local_ptr,
        byte_offset,
        num_elements,
        reduce_op,
        dtype,
        _semantic=None,
    ):
        """
        NVSHMEM-specific all-reduce operation.

        DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
        NOT efficient. It is provided solely to demonstrate the symmetric memory
        abstraction layer API. This implementation should NOT be used as a reference
        for production kernels and is NOT part of the proposed set of kernels that
        constitute the symmetric memory abstraction layer.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr, byte_offset, num_elements, reduce_op, dtype],
            {
                # C function signature: (int64, int64, int32, int32, int32, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int32"),  # byte_offset
                    core.dtype("int32"),  # num_elements
                    core.dtype("int32"),  # reduce_op
                    core.dtype("int32"),  # dtype
                ): ("nvshmem_symm_all_reduce", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @triton.jit
    def symm_all_reduce(
        ctx_ptr,
        local_ptr,
        byte_offset,
        num_elements,
        reduce_op: tl.constexpr,
        dtype: tl.constexpr,
        backend: tl.constexpr = 0,
    ):
        """
        Perform unified all-reduce operation on symmetric memory buffers.

        DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
        NOT efficient. It is provided solely to demonstrate the symmetric memory
        abstraction layer API. This implementation should NOT be used as a reference
        for production kernels and is NOT part of the proposed set of kernels that
        constitute the symmetric memory abstraction layer.

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        This is a collective operation that must be called by all ranks/PEs.
        The context, offset, and num_elements should be the same across all ranks.

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            local_ptr: Pointer to local buffer (device pointer as int64)
            byte_offset: Byte offset within symmetric buffer (int32)
            num_elements: Number of elements to reduce (int32)
            reduce_op: Reduction operation (constexpr)
                       - 0 (REDUCE_OP_SUM): Sum reduction (only supported value)
            dtype: Data type (constexpr)
                   - 0 (DTYPE_FLOAT32): float32 (only supported value)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int32: 0 on success, negative on error
                   -1: null context or invalid context type
                   -2: unknown context type (only for frontend dispatch)
                   -3: unsupported reduction operation
                   -4: unsupported data type

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        # Validate reduce_op at compile time
        tl.static_assert(
            reduce_op == 0,  # REDUCE_OP_SUM
            "Only REDUCE_OP_SUM (0) is supported",
        )

        # Validate dtype at compile time
        tl.static_assert(
            dtype == 0,  # DTYPE_FLOAT32
            "Only DTYPE_FLOAT32 (0) is supported",
        )

        # Use integer literals for comparison since Triton can't access globals
        # 0 = BACKEND_DEFAULT, 1 = BACKEND_NCCL, 2 = BACKEND_NVSHMEM
        if backend == 0:  # BACKEND_DEFAULT
            # Runtime dispatch based on SymmContext type
            return _symm_all_reduce_frontend(
                ctx_ptr,
                local_ptr,
                byte_offset,
                num_elements,
                reduce_op,
                dtype,
            )
        elif backend == 2:  # BACKEND_NVSHMEM
            # Direct NVSHMEM dispatch
            return _nvshmem_symm_all_reduce(
                ctx_ptr,
                local_ptr,
                byte_offset,
                num_elements,
                reduce_op,
                dtype,
            )
        else:
            # BACKEND_NCCL (1) or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )
            return -1

else:
    # Triton not available - provide stubs

    def requires_torch_symm(jit_func):  # type: ignore[misc]
        """Stub for when Triton is not available."""
        raise ImportError("Triton is required for requires_torch_symm decorator")

    def symm_all_reduce(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_all_reduce")


__all__ = [
    # Backend hint constants
    "BACKEND_DEFAULT",
    "BACKEND_NCCL",
    "BACKEND_NVSHMEM",
    # Reduction operation constants
    "REDUCE_OP_SUM",
    # Data type constants
    "DTYPE_FLOAT32",
    # Library finder
    "TorchSymmLibFinder",
    # Decorators
    "requires_torch_symm",
    # Triton extern function
    "symm_all_reduce",
]
