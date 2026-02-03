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
            symm_all_reduce(
                ctx_ptr,
                buffer_ptr,
                byte_offset,
                n_elems,
                REDUCE_OP_SUM,
                DTYPE_FLOAT32,
                backend_hint,
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

# Fence scope constants (matching CUDA constants)
_FENCE_SCOPE_CTA = 0  # Intra-block sync (__syncthreads())
_FENCE_SCOPE_GPU = 1  # Device-wide memory fence (__threadfence())
_FENCE_SCOPE_SYSTEM = 2  # System-wide fence (__threadfence_system())

FENCE_SCOPE_CTA = _FENCE_SCOPE_CTA
FENCE_SCOPE_GPU = _FENCE_SCOPE_GPU
FENCE_SCOPE_SYSTEM = _FENCE_SCOPE_SYSTEM

# Signal operation constants (matching CUDA constants)
_SIGNAL_OP_SET = 0  # Atomic set (replace value)
_SIGNAL_OP_ADD = 1  # Atomic add (increment value)

SIGNAL_OP_SET = _SIGNAL_OP_SET
SIGNAL_OP_ADD = _SIGNAL_OP_ADD

# Signal comparison condition constants (matching NVSHMEM constants)
# These are used with symm_signal_wait_until
_SIGNAL_CMP_EQ = 1  # Equal
_SIGNAL_CMP_NE = 2  # Not equal
_SIGNAL_CMP_GT = 3  # Greater than
_SIGNAL_CMP_GE = 4  # Greater than or equal
_SIGNAL_CMP_LT = 5  # Less than
_SIGNAL_CMP_LE = 6  # Less than or equal

SIGNAL_CMP_EQ = _SIGNAL_CMP_EQ
SIGNAL_CMP_NE = _SIGNAL_CMP_NE
SIGNAL_CMP_GT = _SIGNAL_CMP_GT
SIGNAL_CMP_GE = _SIGNAL_CMP_GE
SIGNAL_CMP_LT = _SIGNAL_CMP_LT
SIGNAL_CMP_LE = _SIGNAL_CMP_LE

# Barrier scope constants
# These control which peers participate in barrier synchronization
_SCOPE_LSA = 0  # LSA (Local Symmetric Access) domain only - NVLink-connected peers
_SCOPE_WORLD = 1  # All ranks in the team (full world synchronization via GIN)

SCOPE_LSA = _SCOPE_LSA
SCOPE_WORLD = _SCOPE_WORLD

# Triton constexpr versions of constants for use within @triton.jit functions
# These are only defined when Triton is available
# IMPORTANT: Must use tl.constexpr() function, not tl.constexpr type annotation
if TRITON_AVAILABLE:
    # Backend constants
    TL_BACKEND_DEFAULT = tl.constexpr(_BACKEND_DEFAULT)
    TL_BACKEND_NCCL = tl.constexpr(_BACKEND_NCCL)
    TL_BACKEND_NVSHMEM = tl.constexpr(_BACKEND_NVSHMEM)

    # Reduction operation constants
    TL_REDUCE_OP_SUM = tl.constexpr(_REDUCE_OP_SUM)

    # Data type constants
    TL_DTYPE_FLOAT32 = tl.constexpr(_DTYPE_FLOAT32)

    # Fence scope constants
    TL_FENCE_SCOPE_CTA = tl.constexpr(_FENCE_SCOPE_CTA)
    TL_FENCE_SCOPE_GPU = tl.constexpr(_FENCE_SCOPE_GPU)
    TL_FENCE_SCOPE_SYSTEM = tl.constexpr(_FENCE_SCOPE_SYSTEM)

    # Signal operation constants
    TL_SIGNAL_OP_SET = tl.constexpr(_SIGNAL_OP_SET)
    TL_SIGNAL_OP_ADD = tl.constexpr(_SIGNAL_OP_ADD)

    # Signal comparison constants
    TL_SIGNAL_CMP_EQ = tl.constexpr(_SIGNAL_CMP_EQ)
    TL_SIGNAL_CMP_NE = tl.constexpr(_SIGNAL_CMP_NE)
    TL_SIGNAL_CMP_GT = tl.constexpr(_SIGNAL_CMP_GT)
    TL_SIGNAL_CMP_GE = tl.constexpr(_SIGNAL_CMP_GE)
    TL_SIGNAL_CMP_LT = tl.constexpr(_SIGNAL_CMP_LT)
    TL_SIGNAL_CMP_LE = tl.constexpr(_SIGNAL_CMP_LE)

    # Scope constants
    TL_SCOPE_LSA = tl.constexpr(_SCOPE_LSA)
    TL_SCOPE_WORLD = tl.constexpr(_SCOPE_WORLD)


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
                    symm_all_reduce(
                        ctx_ptr,
                        buffer_ptr,
                        byte_offset,
                        n_elems,
                        REDUCE_OP_SUM,
                        DTYPE_FLOAT32,
                        backend_hint,
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

        Asserts on invalid context or unsupported reduce_op/dtype.
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

        Asserts on invalid context or unsupported reduce_op/dtype.
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

    # =========================================================================
    # SYMM_QUIET - FLUSH/COMPLETE ALL OUTSTANDING SYMMETRIC OPERATIONS
    # =========================================================================

    @core.extern
    def _symm_quiet_frontend(
        ctx_ptr,
        _semantic=None,
    ):
        """
        Frontend quiet operation that dispatches based on SymmContext type.

        Flushes/completes all outstanding symmetric operations issued by this rank.
        Ensures that all prior symm_put/_signal calls are completed (data delivered
        to destination) before proceeding.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): ("symm_quiet", core.dtype("int32")),
            },
            is_pure=False,  # Has side effects (memory ordering)
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_quiet(
        ctx_ptr,
        _semantic=None,
    ):
        """
        NVSHMEM-specific quiet operation.

        Flushes/completes all outstanding NVSHMEM operations issued by this PE.
        Maps to nvshmem_quiet().

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): (
                    "nvshmem_symm_quiet",
                    core.dtype("int32"),
                ),
            },
            is_pure=False,  # Has side effects (memory ordering)
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_BARRIER - TEAM-WIDE BARRIER SYNCHRONIZATION
    # =========================================================================

    @core.extern
    def _symm_barrier_frontend(
        ctx_ptr,
        _semantic=None,
    ):
        """
        Frontend barrier operation that dispatches based on SymmContext type.

        Performs a team-wide barrier synchronization. All ranks in the team block
        until everyone has reached this point.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): (
                    "symm_barrier",
                    core.dtype("int32"),
                ),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_barrier(
        ctx_ptr,
        _semantic=None,
    ):
        """
        NVSHMEM-specific barrier operation.

        Performs a team-wide barrier synchronization using NVSHMEM.
        Maps to nvshmemx_barrier_all_block().

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): (
                    "nvshmem_symm_barrier",
                    core.dtype("int32"),
                ),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_LSA_BARRIER - LSA DOMAIN BARRIER SYNCHRONIZATION
    # =========================================================================

    @core.extern
    def _symm_lsa_barrier_frontend(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        Frontend LSA barrier operation that dispatches based on SymmContext type.

        Performs barrier synchronization among ranks in the same LSA (Local
        Symmetric Access) domain. Only ranks that can directly access each
        other's memory via load/store operations participate in this barrier.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        For NVSHMEM, uses nvshmemx_team_barrier_block with the LSA team.
        For NCCL, uses ncclLsaBarrierSession with ncclTeamTagLsa.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("symm_lsa_barrier", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_lsa_barrier(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        NVSHMEM-specific LSA barrier operation.

        Performs barrier synchronization among ranks in the same LSA (Local
        Symmetric Access) domain using NVSHMEM.
        Maps to nvshmemx_team_barrier_block(lsa_team).

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("nvshmem_symm_lsa_barrier", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_LSA_BARRIER_ARRIVE - LSA DOMAIN BARRIER ARRIVE (SPLIT-PHASE)
    # =========================================================================

    @core.extern
    def _symm_lsa_barrier_arrive_frontend(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        Frontend LSA barrier arrive operation that dispatches based on SymmContext type.

        Signals arrival at an LSA barrier without waiting for other peers.
        This is the "arrive" phase of a split-phase barrier, allowing overlapping
        computation while waiting for peers.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        For NCCL, uses ncclLsaBarrierSession with its arrive() method.
        For NVSHMEM, uses signal operations on the LSA signal pad.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("symm_lsa_barrier_arrive", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_lsa_barrier_arrive(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        NVSHMEM-specific LSA barrier arrive operation.

        Signals arrival at an LSA barrier without waiting for other peers.
        This is the "arrive" phase of a split-phase barrier.

        Uses signal operations on the LSA signal pad to notify all peers of
        arrival. Each rank atomically increments a designated barrier signal
        on all peer signal pads.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("nvshmem_symm_lsa_barrier_arrive", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_LSA_BARRIER_WAIT - LSA DOMAIN BARRIER WAIT (SPLIT-PHASE)
    # =========================================================================

    @core.extern
    def _symm_lsa_barrier_wait_frontend(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        Frontend LSA barrier wait operation that dispatches based on SymmContext type.

        Waits for all peers to arrive at the LSA barrier.
        This is the "wait" phase of a split-phase barrier.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        For NCCL, uses ncclLsaBarrierSession with its wait() method.
        For NVSHMEM, waits until the local barrier signal reaches the expected
        count, then resets the signal for the next iteration.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("symm_lsa_barrier_wait", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_lsa_barrier_wait(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        NVSHMEM-specific LSA barrier wait operation.

        Waits for all peers to arrive at the LSA barrier.
        This is the "wait" phase of a split-phase barrier.

        Waits until the local barrier signal (at lsa_signal_pad[barrier_index])
        reaches the expected count (world_size), indicating all peers have called
        symm_lsa_barrier_arrive. After wait completes, the barrier signal
        is automatically reset to 0 for the next iteration.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("nvshmem_symm_lsa_barrier_wait", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_BARRIER_ARRIVE - GIN BARRIER ARRIVE (SPLIT-PHASE, FULL TEAM)
    # =========================================================================

    @core.extern
    def _symm_barrier_arrive_frontend(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        Frontend GIN barrier arrive operation that dispatches based on SymmContext type.

        Signals arrival at a GIN barrier without waiting for other peers.
        This is the "arrive" phase of a split-phase barrier, allowing overlapping
        computation while waiting for peers.

        Unlike LSA barrier which only targets the LSA domain, this GIN barrier
        targets ALL ranks in the team using GPU-initiated networking.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        For NCCL, uses ncclLsaBarrierSession with ncclTeamTagWorld.
        For NVSHMEM, uses signal operations on the GIN signal pad.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("symm_barrier_arrive", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_barrier_arrive(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        NVSHMEM-specific GIN barrier arrive operation.

        Signals arrival at a GIN barrier without waiting for other peers.
        This is the "arrive" phase of a split-phase barrier.

        Uses signal operations on the GIN signal pad to notify all peers of
        arrival. Each rank atomically increments a designated barrier signal
        on all peer signal pads.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("nvshmem_symm_barrier_arrive", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_BARRIER_WAIT - GIN BARRIER WAIT (SPLIT-PHASE, FULL TEAM)
    # =========================================================================

    @core.extern
    def _symm_barrier_wait_frontend(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        Frontend GIN barrier wait operation that dispatches based on SymmContext type.

        Waits for all peers to arrive at the GIN barrier.
        This is the "wait" phase of a split-phase barrier.

        Unlike LSA barrier which only waits for LSA domain peers, this GIN barrier
        waits for ALL ranks in the team.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        For NCCL, uses ncclLsaBarrierSession with ncclTeamTagWorld.
        For NVSHMEM, waits until the local GIN barrier signal reaches the expected
        count, then updates the epoch for the next iteration.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("symm_barrier_wait", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_barrier_wait(
        ctx_ptr,
        barrier_index,
        _semantic=None,
    ):
        """
        NVSHMEM-specific GIN barrier wait operation.

        Waits for all peers to arrive at the GIN barrier.
        This is the "wait" phase of a split-phase barrier.

        Waits until the local GIN barrier signal (at gin_signal_pad[barrier_index])
        reaches the expected count (team_size), indicating all peers have called
        symm_barrier_arrive. After wait completes, the epoch is updated for
        the next iteration.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, barrier_index],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # barrier_index
                ): ("nvshmem_symm_barrier_wait", core.dtype("int32")),
            },
            is_pure=False,  # Collective operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_FENCE - MEMORY FENCE FOR ORDERING
    # =========================================================================

    @core.extern
    def _symm_fence_frontend(
        ctx_ptr,
        scope,
        _semantic=None,
    ):
        """
        Frontend fence operation that dispatches based on SymmContext type.

        Provides memory ordering guarantees at different scopes and also
        inserts appropriate NVSHMEM/NIC fence for symmetric operations.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context or invalid scope.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, scope],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # scope
                ): ("symm_fence", core.dtype("int32")),
            },
            is_pure=False,  # Has side effects (memory ordering)
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_fence(
        ctx_ptr,
        scope,
        _semantic=None,
    ):
        """
        NVSHMEM-specific fence operation.

        Provides memory ordering guarantees at different scopes:
        - FENCE_SCOPE_CTA (0): __syncthreads() + nvshmem_fence()
        - FENCE_SCOPE_GPU (1): __threadfence() + nvshmem_fence()
        - FENCE_SCOPE_SYSTEM (2): __threadfence_system() + nvshmem_fence()

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context or invalid scope.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, scope],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # scope
                ): ("nvshmem_symm_fence", core.dtype("int32")),
            },
            is_pure=False,  # Has side effects (memory ordering)
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

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context, unsupported reduce_op, or
              unsupported dtype. Use device-side error checking for debugging.
        """
        # Validate reduce_op at compile time
        tl.static_assert(
            reduce_op == TL_REDUCE_OP_SUM,
            "Only REDUCE_OP_SUM (0) is supported",
        )

        # Validate dtype at compile time
        tl.static_assert(
            dtype == TL_DTYPE_FLOAT32,
            "Only DTYPE_FLOAT32 (0) is supported",
        )

        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_all_reduce_frontend(
                ctx_ptr,
                local_ptr,
                byte_offset,
                num_elements,
                reduce_op,
                dtype,
            )
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_all_reduce(
                ctx_ptr,
                local_ptr,
                byte_offset,
                num_elements,
                reduce_op,
                dtype,
            )
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

    @triton.jit
    def _symm_lsa_signal_ptr_frontend(
        ctx_ptr,
        peer,
    ):
        """
        [INTERNAL] Unified frontend for lsa_signal_ptr operation.
        Uses runtime dispatch based on SymmContext type.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, peer],
            {
                # C function signature:
                # (int64 ctx_ptr, int32 peer) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # peer
                ): ("symm_lsa_signal_ptr", core.dtype("int64")),
            },
            is_pure=True,  # Returns pointer without side effects
            _semantic=_semantic,
        )

    @triton.jit
    def _nvshmem_symm_lsa_signal_ptr(
        ctx_ptr,
        peer,
    ):
        """
        [INTERNAL] NVSHMEM-specific lsa_signal_ptr implementation.

        Get a device pointer to peer's LSA signal pad (if accessible via P2P).
        Uses nvshmem_ptr() to get the remote address.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Returns 0 if peer's signal pad is not accessible via P2P.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, peer],
            {
                # C function signature:
                # (int64 ctx_ptr, int32 peer) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # peer
                ): ("nvshmem_symm_lsa_signal_ptr", core.dtype("int64")),
            },
            is_pure=True,  # Returns pointer without side effects
            _semantic=_semantic,
        )

    @triton.jit
    def symm_lsa_signal_ptr(
        ctx_ptr,
        peer,
        backend: tl.constexpr = 0,
    ):
        """
        Get a device pointer to peer's signal pad (if accessible via P2P/LSA).

        Returns a device pointer to the peer's signal pad that can be used for
        direct load/store operations. This is useful for implementing custom
        signaling patterns with direct memory access.

        For NVSHMEM, this returns the LSA signal pad (used for P2P load/store).
        The GIN signal pad (used by symm_signal for atomic operations) is
        separate and not accessible via this function.

        For NCCL, this returns the signal pad accessible via the LSA window.

        If the peer's signal pad is not accessible via P2P (e.g., remote node),
        returns 0/None.

        Example usage:
            # Get pointer to peer's signal pad
            peer_signal_pad = symm_lsa_signal_ptr(ctx_ptr, peer)

            # If accessible, can use direct load/store
            if peer_signal_pad != 0:
                # Direct store to peer's signal (index 0)
                tl.store(peer_signal_pad, value)

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            peer: Peer rank/PE to get signal pad pointer for (int32)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            Device pointer to peer's signal pad (int64), or 0 if not P2P accessible.

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            return _symm_lsa_signal_ptr_frontend(ctx_ptr, peer)
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            return _nvshmem_symm_lsa_signal_ptr(ctx_ptr, peer)
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )
            return tl.zeros((1,), dtype=tl.int64)[0]  # Unreachable

    @triton.jit
    def symm_quiet(
        ctx_ptr,
        backend: tl.constexpr = 0,
    ):
        """
        Flush/complete all outstanding symmetric operations issued by this rank.

        This ensures that all prior symm_put/_signal calls are completed (data
        delivered to destination) before proceeding. This is a local completion
        guarantee - it does not synchronize with other ranks.

        Use this after a series of puts/signals to ensure all data has been
        delivered before proceeding to dependent operations.

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmem_quiet()
        - NCCL: gin.flush() (when device bitcode is available)

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_quiet_frontend(ctx_ptr)
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_quiet(ctx_ptr)
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

    @triton.jit
    def symm_barrier(
        ctx_ptr,
        scope: tl.constexpr = 0,
        backend: tl.constexpr = 0,
    ):
        """
        Perform barrier synchronization with configurable scope.

        All ranks in the specified scope block until everyone has reached this point.
        This is a collective operation that must be called by all ranks in the scope.

        The scope parameter controls which peers participate:
        - SCOPE_LSA (0): Only LSA domain peers (NVLink-connected)
        - SCOPE_WORLD (1): All ranks in the team (via GIN)

        Unlike symm_quiet (which ensures local operations are complete), symm_barrier
        provides global synchronization where all ranks wait for each other.

        Common usage pattern:
          1. All ranks perform local operations (puts, signals, etc.)
          2. Call symm_barrier to ensure all ranks have completed their operations
          3. Proceed with operations that depend on data from other ranks

        Example usage:
            # LSA-only barrier (faster, NVLink peers only)
            symm_barrier(ctx_ptr, scope=SCOPE_LSA, backend=backend_hint)

            # Full team barrier (all ranks, including cross-node)
            symm_barrier(ctx_ptr, scope=SCOPE_WORLD, backend=backend_hint)

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to (when scope=SCOPE_LSA):
        - NVSHMEM: Signal-based LSA barrier using lsa_signal_pad
        - NCCL: ncclLsaBarrier() with ncclTeamTagLsa (when device bitcode is available)

        Maps to (when scope=SCOPE_WORLD):
        - NVSHMEM: nvshmemx_barrier_all_block()
        - NCCL: ncclLsaBarrier() with ncclTeamTagWorld (when device bitcode is available)

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            scope: Barrier scope (constexpr, default=0 for SCOPE_LSA)
                   - 0 (SCOPE_LSA): LSA domain only (NVLink-connected peers)
                   - 1 (SCOPE_WORLD): All ranks in the team
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        # Validate scope at compile time
        tl.static_assert(
            scope >= TL_SCOPE_LSA and scope <= TL_SCOPE_WORLD,
            "scope must be 0 (SCOPE_LSA) or 1 (SCOPE_WORLD)",
        )

        # Dispatch based on scope
        if scope == TL_SCOPE_LSA:
            if backend == TL_BACKEND_DEFAULT:
                _symm_lsa_barrier_frontend(ctx_ptr, 0)  # barrier_index=0
            elif backend == TL_BACKEND_NVSHMEM:
                _nvshmem_symm_lsa_barrier(ctx_ptr, 0)  # barrier_index=0
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
        else:  # SCOPE_WORLD
            if backend == TL_BACKEND_DEFAULT:
                _symm_barrier_frontend(ctx_ptr)
            elif backend == TL_BACKEND_NVSHMEM:
                _nvshmem_symm_barrier(ctx_ptr)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )

    # =========================================================================
    # UNIFIED BARRIER PRIMITIVES WITH SCOPE
    # These provide a unified interface with scope parameter to select between
    # LSA (local) and WORLD (all ranks) synchronization.
    # =========================================================================

    @triton.jit
    def symm_barrier_arrive(
        ctx_ptr,
        barrier_index: tl.constexpr = 0,
        scope: tl.constexpr = 0,
        backend: tl.constexpr = 0,
    ):
        """
        Signal arrival at a barrier with configurable scope.

        This is the "arrive" phase of a split-phase barrier. It signals that this
        rank has completed its local operations and is ready to synchronize, but
        does NOT wait for other ranks to arrive.

        The scope parameter controls which peers participate:
        - SCOPE_LSA (0): Only LSA domain peers (NVLink-connected)
        - SCOPE_WORLD (1): All ranks in the team (via GIN)

        Split-phase barriers allow overlapping computation with synchronization:
        1. All ranks call symm_barrier_arrive() after completing local work
        2. Ranks can perform independent computation
        3. All ranks call symm_barrier_wait() before accessing peer data

        Multiple independent barriers can be used by specifying different barrier_index
        values. The same barrier_index must be used in the corresponding wait() call.

        Example usage:
            # Phase 1: Write data to local buffer
            tl.store(local_ptr + offsets, data, mask=mask)

            # Signal arrival (data is ready) - LSA scope
            symm_barrier_arrive(ctx_ptr, barrier_index=0, scope=SCOPE_LSA)

            # Do independent computation while waiting for peers
            result = compute_something_else()

            # Wait for all LSA peers before reading their data
            symm_barrier_wait(ctx_ptr, barrier_index=0, scope=SCOPE_LSA)

            # Now safe to read peer data
            peer_data = tl.load(peer_ptr + offsets, mask=mask)

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to (when scope=SCOPE_LSA):
        - NVSHMEM: Signal operations on lsa_signal_pad to notify peers
        - NCCL: ncclLsaBarrierSession::arrive() with ncclTeamTagLsa

        Maps to (when scope=SCOPE_WORLD):
        - NVSHMEM: Signal operations on gin_signal_pad to notify all team peers
        - NCCL: ncclLsaBarrierSession::arrive() with ncclTeamTagWorld

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            barrier_index: Index of the barrier to use (constexpr, default=0).
                           Multiple independent barriers can be used by specifying
                           different indices. Must match the corresponding wait() call.
            scope: Barrier scope (constexpr, default=0 for SCOPE_LSA)
                   - 0 (SCOPE_LSA): LSA domain only (NVLink-connected peers)
                   - 1 (SCOPE_WORLD): All ranks in the team
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        # Validate scope at compile time
        tl.static_assert(
            scope >= TL_SCOPE_LSA and scope <= TL_SCOPE_WORLD,
            "scope must be 0 (SCOPE_LSA) or 1 (SCOPE_WORLD)",
        )

        # Dispatch based on scope
        if scope == TL_SCOPE_LSA:
            if backend == TL_BACKEND_DEFAULT:
                _symm_lsa_barrier_arrive_frontend(ctx_ptr, barrier_index)
            elif backend == TL_BACKEND_NVSHMEM:
                _nvshmem_symm_lsa_barrier_arrive(ctx_ptr, barrier_index)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
        else:  # SCOPE_WORLD
            if backend == TL_BACKEND_DEFAULT:
                _symm_barrier_arrive_frontend(ctx_ptr, barrier_index)
            elif backend == TL_BACKEND_NVSHMEM:
                _nvshmem_symm_barrier_arrive(ctx_ptr, barrier_index)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )

    @triton.jit
    def symm_barrier_wait(
        ctx_ptr,
        barrier_index: tl.constexpr = 0,
        scope: tl.constexpr = 0,
        backend: tl.constexpr = 0,
    ):
        """
        Wait for all peers to arrive at a barrier with configurable scope.

        This is the "wait" phase of a split-phase barrier. It blocks until all ranks
        in the specified scope have called symm_barrier_arrive() with the
        same barrier_index and scope.

        The scope parameter controls which peers we wait for:
        - SCOPE_LSA (0): Only LSA domain peers (NVLink-connected)
        - SCOPE_WORLD (1): All ranks in the team (via GIN)

        Must be called after symm_barrier_arrive() with the same barrier_index
        and scope. After this returns:
        - All data written by peers before their arrive() is guaranteed to be visible
        - The barrier state is automatically reset for the next iteration

        Split-phase barriers allow overlapping computation with synchronization:
        1. All ranks call symm_barrier_arrive() after completing local work
        2. Ranks can perform independent computation
        3. All ranks call symm_barrier_wait() before accessing peer data

        Example usage:
            # Phase 1: Write data to local buffer
            tl.store(local_ptr + offsets, data, mask=mask)

            # Signal arrival (data is ready) - WORLD scope
            symm_barrier_arrive(ctx_ptr, barrier_index=0, scope=SCOPE_WORLD)

            # Do independent computation while waiting for peers
            result = compute_something_else()

            # Wait for all peers before reading their data
            symm_barrier_wait(ctx_ptr, barrier_index=0, scope=SCOPE_WORLD)

            # Now safe to read peer data
            peer_data = tl.load(peer_ptr + offsets, mask=mask)

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to (when scope=SCOPE_LSA):
        - NVSHMEM: nvshmem_signal_wait_until on lsa_signal_pad, then reset
        - NCCL: ncclLsaBarrierSession::wait() with ncclTeamTagLsa

        Maps to (when scope=SCOPE_WORLD):
        - NVSHMEM: nvshmem_signal_wait_until on gin_signal_pad, then update epoch
        - NCCL: ncclLsaBarrierSession::wait() with ncclTeamTagWorld

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            barrier_index: Index of the barrier to wait on (constexpr, default=0).
                           Must match the barrier_index used in the corresponding
                           arrive() call.
            scope: Barrier scope (constexpr, default=0 for SCOPE_LSA)
                   - 0 (SCOPE_LSA): LSA domain only (NVLink-connected peers)
                   - 1 (SCOPE_WORLD): All ranks in the team
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        # Validate scope at compile time
        tl.static_assert(
            scope >= TL_SCOPE_LSA and scope <= TL_SCOPE_WORLD,
            "scope must be 0 (SCOPE_LSA) or 1 (SCOPE_WORLD)",
        )

        # Dispatch based on scope
        if scope == TL_SCOPE_LSA:
            if backend == TL_BACKEND_DEFAULT:
                _symm_lsa_barrier_wait_frontend(ctx_ptr, barrier_index)
            elif backend == TL_BACKEND_NVSHMEM:
                _nvshmem_symm_lsa_barrier_wait(ctx_ptr, barrier_index)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
        else:  # SCOPE_WORLD
            if backend == TL_BACKEND_DEFAULT:
                _symm_barrier_wait_frontend(ctx_ptr, barrier_index)
            elif backend == TL_BACKEND_NVSHMEM:
                _nvshmem_symm_barrier_wait(ctx_ptr, barrier_index)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )

    @triton.jit
    def symm_fence(
        ctx_ptr,
        scope: tl.constexpr,
        backend: tl.constexpr = 0,
    ):
        """
        Memory fence for ordering at different scopes.

        Provides memory ordering guarantees at different scopes:
        - FENCE_SCOPE_CTA (0): Intra-block sync (__syncthreads())
          Synchronizes all threads within the same thread block.
        - FENCE_SCOPE_GPU (1): Device-wide memory fence (__threadfence())
          Ensures all prior memory writes are visible to all threads on the GPU.
        - FENCE_SCOPE_SYSTEM (2): System-wide fence (__threadfence_system())
          Ensures all prior memory writes are visible system-wide (CPU, other GPUs).

        For symmetric operations, this also inserts appropriate NVSHMEM/NIC fence
        to enforce ordering between consecutive puts. This ensures that if you do:
          symm_put(dest, data1, pe)
          symm_fence(ctx, FENCE_SCOPE_GPU)
          symm_put(dest, data2, pe)
        The first put will be ordered before the second put to the same PE.

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: __syncthreads()/__threadfence()/__threadfence_system() + nvshmem_fence()
        - NCCL: __syncthreads()/__threadfence()/__threadfence_system() (when device bitcode available)

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            scope: Fence scope (constexpr)
                   - 0 (FENCE_SCOPE_CTA): Intra-block synchronization
                   - 1 (FENCE_SCOPE_GPU): Device-wide memory fence
                   - 2 (FENCE_SCOPE_SYSTEM): System-wide memory fence
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context or invalid scope.
        """
        # Validate scope at compile time
        tl.static_assert(
            scope >= TL_FENCE_SCOPE_CTA and scope <= TL_FENCE_SCOPE_SYSTEM,
            "scope must be 0 (CTA), 1 (GPU), or 2 (SYSTEM)",
        )

        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_fence_frontend(ctx_ptr, scope)
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_fence(ctx_ptr, scope)
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

    # =========================================================================
    # SYMM_LSA_PTR - GET DEVICE POINTER TO PEER'S SYMMETRIC BUFFER
    # =========================================================================

    @core.extern
    def _symm_lsa_ptr_frontend(
        ctx_ptr,
        local_ptr,
        peer,
        _semantic=None,
    ):
        """
        Frontend LSA pointer operation that dispatches based on SymmContext type.

        Gets a device pointer to the peer's symmetric buffer. This pointer can be
        used for direct P2P access via tl.load/tl.store if the peer is accessible
        via P2P (e.g., NVLink connected GPUs).

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr, peer],
            {
                # C function signature: (int64, int64, int32) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int32"),  # peer
                ): ("symm_lsa_ptr", core.dtype("int64")),
            },
            is_pure=True,  # Pure function - just returns a pointer
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_lsa_ptr(
        ctx_ptr,
        local_ptr,
        peer,
        _semantic=None,
    ):
        """
        NVSHMEM-specific LSA pointer operation.

        Gets a device pointer to the peer's symmetric buffer using NVSHMEM.
        Maps to nvshmem_ptr().

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr, peer],
            {
                # C function signature: (int64, int64, int32) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                    core.dtype("int32"),  # peer
                ): ("nvshmem_symm_lsa_ptr", core.dtype("int64")),
            },
            is_pure=True,  # Pure function - just returns a pointer
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_LSA_MULTICAST_PTR - GET MULTICAST POINTER FOR BROADCASTING
    # =========================================================================

    @core.extern
    def _symm_lsa_multicast_ptr_frontend(
        ctx_ptr,
        local_ptr,
        _semantic=None,
    ):
        """
        Frontend LSA multicast pointer operation that dispatches based on SymmContext type.

        Gets a multicast pointer for broadcasting to all peers. This is used for
        efficient one-to-many communication where the same data is sent to all
        peers in the group.

        Returns 0 if multicast is not supported by the hardware.

        Team is obtained from the context internally.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr],
            {
                # C function signature: (int64, int64) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                ): ("symm_lsa_multicast_ptr", core.dtype("int64")),
            },
            is_pure=True,  # Pure function - just returns a pointer
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_lsa_multicast_ptr(
        ctx_ptr,
        local_ptr,
        _semantic=None,
    ):
        """
        NVSHMEM-specific LSA multicast pointer operation.

        Gets a multicast pointer for broadcasting to all peers using NVSHMEM.
        Requires hardware support (e.g., NVSwitch with multicast capability).

        Returns 0 if multicast is not supported.

        Team is obtained from the context internally.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, local_ptr],
            {
                # C function signature: (int64, int64) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # local_ptr
                ): ("nvshmem_symm_lsa_multicast_ptr", core.dtype("int64")),
            },
            is_pure=True,  # Pure function - just returns a pointer
            _semantic=_semantic,
        )

    @triton.jit
    def symm_lsa_ptr(
        ctx_ptr,
        local_ptr,
        peer,
        backend: tl.constexpr = 0,
    ):
        """
        Get a device pointer to peer's symmetric buffer for direct P2P/LSA access.

        This function returns a device pointer that can be used for direct
        tl.load/tl.store operations to access the peer's symmetric buffer.
        If the peer is not accessible via P2P (e.g., not NVLink connected),
        returns 0 (null pointer).

        Usage pattern:
          peer_ptr = symm_lsa_ptr(ctx, my_local_ptr, peer_rank)
          if peer_ptr != 0:
              data = tl.load(peer_ptr + offset)  # Direct P2P read
              tl.store(peer_ptr + offset, value)  # Direct P2P write

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmem_ptr(local_ptr, peer)
        - NCCL: ncclGetLsaPointer() (when device bitcode is available)

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            local_ptr: Pointer to local symmetric buffer (device pointer as int64)
            peer: Peer rank/PE to get pointer for (int32)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int64: Device pointer to peer's symmetric buffer, or 0 if not accessible

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            return _symm_lsa_ptr_frontend(ctx_ptr, local_ptr, peer)
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            return _nvshmem_symm_lsa_ptr(ctx_ptr, local_ptr, peer)
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )
            return 0

    @triton.jit
    def symm_lsa_multicast_ptr(
        ctx_ptr,
        local_ptr,
        backend: tl.constexpr = 0,
    ):
        """
        Get a multicast pointer for broadcasting to all peers via LSA.

        This function returns a multicast pointer that can be used for efficient
        one-to-many communication. When data is stored to this pointer, it is
        automatically broadcast to all peers in the group.

        Returns 0 if multicast is not supported by the hardware (e.g., no NVSwitch
        with multicast capability).

        Usage pattern:
          mc_ptr = symm_lsa_multicast_ptr(ctx, my_local_ptr)
          if mc_ptr != 0:
              tl.store(mc_ptr + offset, value)  # Broadcast to all peers

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Team is obtained from the context internally.

        Maps to:
        - NVSHMEM: nvshmemx_mc_ptr(team, ptr) where team is retrieved from context
        - NCCL: ncclGetLsaMultimemPointer() (when device bitcode is available)

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            local_ptr: Pointer to local symmetric buffer (device pointer as int64)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int64: Multicast pointer for broadcasting, or 0 if not supported

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            return _symm_lsa_multicast_ptr_frontend(ctx_ptr, local_ptr)
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            return _nvshmem_symm_lsa_multicast_ptr(ctx_ptr, local_ptr)
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )
            return 0

    # =========================================================================
    # SYMM_TEAM PRIMITIVES - TOPOLOGY MANAGEMENT
    # =========================================================================

    @core.extern
    def _symm_team_size_frontend(
        ctx_ptr,
        _semantic=None,
    ):
        """
        Frontend team_size operation.

        Returns the number of ranks in the team.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): ("symm_team_size", core.dtype("int32")),
            },
            is_pure=True,  # Pure function - just returns team size
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_team_size(
        ctx_ptr,
        _semantic=None,
    ):
        """
        NVSHMEM-specific team_size operation.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): ("nvshmem_symm_team_size", core.dtype("int32")),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def _symm_team_rank_frontend(
        ctx_ptr,
        _semantic=None,
    ):
        """
        Frontend team_rank operation.

        Returns the calling process's rank index within the team.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): ("symm_team_rank", core.dtype("int32")),
            },
            is_pure=True,  # Pure function - just returns rank
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_team_rank(
        ctx_ptr,
        _semantic=None,
    ):
        """
        NVSHMEM-specific team_rank operation.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): ("nvshmem_symm_team_rank", core.dtype("int32")),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def _symm_team_lsa_size_frontend(
        ctx_ptr,
        _semantic=None,
    ):
        """
        Frontend team_lsa_size operation.

        Returns the number of ranks in the caller's LSA (Local Symmetric Access) domain.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): ("symm_team_lsa_size", core.dtype("int32")),
            },
            is_pure=True,  # Pure function - just returns LSA size
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_team_lsa_size(
        ctx_ptr,
        _semantic=None,
    ):
        """
        NVSHMEM-specific team_lsa_size operation.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr],
            {
                # C function signature: (int64) -> int32
                (core.dtype("int64"),): (
                    "nvshmem_symm_team_lsa_size",
                    core.dtype("int32"),
                ),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def _symm_team_lsa_frontend(
        ctx_ptr,
        peer,
        _semantic=None,
    ):
        """
        Frontend team_lsa operation.

        Returns whether the peer rank is in the same LSA domain as the caller.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, peer],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # peer
                ): ("symm_team_lsa", core.dtype("int32")),
            },
            is_pure=True,  # Pure function - just checks LSA membership
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_team_lsa(
        ctx_ptr,
        peer,
        _semantic=None,
    ):
        """
        NVSHMEM-specific team_lsa operation.
        Team is obtained from the context internally.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, peer],
            {
                # C function signature: (int64, int32) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # peer
                ): ("nvshmem_symm_team_lsa", core.dtype("int32")),
            },
            is_pure=True,
            _semantic=_semantic,
        )

    @triton.jit
    def symm_team_size(
        ctx_ptr,
        scope: tl.constexpr = 1,
        backend: tl.constexpr = 0,
    ):
        """
        Get the number of ranks in the team for the specified scope.

        This returns the total number of processes/PEs that are members of the
        team for the given scope.

        Args:
            ctx_ptr: Pointer to SymmContext (as int64 for Triton compatibility)
            scope: Synchronization scope (constexpr)
                   - 0 (SCOPE_LSA): Return LSA domain size (NVLink-connected peers)
                   - 1 (SCOPE_WORLD): Return full team size (all ranks)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int32: Number of ranks in the team/scope, or -1 if context/team is invalid

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        tl.static_assert(
            scope >= TL_SCOPE_LSA and scope <= TL_SCOPE_WORLD,
            "scope must be 0 (SCOPE_LSA) or 1 (SCOPE_WORLD)",
        )
        if scope == TL_SCOPE_LSA:
            if backend == TL_BACKEND_DEFAULT:
                return _symm_team_lsa_size_frontend(ctx_ptr)
            elif backend == TL_BACKEND_NVSHMEM:
                return _nvshmem_symm_team_lsa_size(ctx_ptr)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
                return -1
        else:  # SCOPE_WORLD
            if backend == TL_BACKEND_DEFAULT:
                return _symm_team_size_frontend(ctx_ptr)
            elif backend == TL_BACKEND_NVSHMEM:
                return _nvshmem_symm_team_size(ctx_ptr)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
                return -1

    @triton.jit
    def symm_team_rank(
        ctx_ptr,
        scope: tl.constexpr = 1,
        backend: tl.constexpr = 0,
    ):
        """
        Get the calling process's rank index within the team for the specified scope.

        Returns the rank of this process within the team.
        For SCOPE_WORLD: returns 0..team_size-1
        For SCOPE_LSA: returns 0..lsa_size-1 (rank within LSA domain)

        Args:
            ctx_ptr: Pointer to SymmContext (as int64 for Triton compatibility)
            scope: Synchronization scope (constexpr)
                   - 0 (SCOPE_LSA): Return rank within LSA domain (0..lsa_size-1)
                   - 1 (SCOPE_WORLD): Return rank within full team (0..team_size-1)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int32: Rank index within the team/scope, or -1 if context/team is invalid

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        tl.static_assert(
            scope >= TL_SCOPE_LSA and scope <= TL_SCOPE_WORLD,
            "scope must be 0 (SCOPE_LSA) or 1 (SCOPE_WORLD)",
        )
        if scope == TL_SCOPE_LSA:
            # Compute LSA rank as world_rank % lsa_size
            if backend == TL_BACKEND_DEFAULT:
                world_rank = _symm_team_rank_frontend(ctx_ptr)
                lsa_size = _symm_team_lsa_size_frontend(ctx_ptr)
                return world_rank % lsa_size
            elif backend == TL_BACKEND_NVSHMEM:
                world_rank = _nvshmem_symm_team_rank(ctx_ptr)
                lsa_size = _nvshmem_symm_team_lsa_size(ctx_ptr)
                return world_rank % lsa_size
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
                return -1
        else:  # SCOPE_WORLD
            if backend == TL_BACKEND_DEFAULT:
                return _symm_team_rank_frontend(ctx_ptr)
            elif backend == TL_BACKEND_NVSHMEM:
                return _nvshmem_symm_team_rank(ctx_ptr)
            else:
                tl.static_assert(
                    False,
                    "NCCL backend not supported (no device bitcode library available)",
                )
                return -1

    @triton.jit
    def symm_team_peer(
        ctx_ptr,
        peer,
        backend: tl.constexpr = 0,
    ):
        """
        Get the scope in which the caller and a peer rank are connected.

        Returns the scope (connectivity level) between the caller and the
        specified peer rank. This can be used to select the optimal
        communication strategy for a given peer.

        Usage pattern:
          scope = symm_team_peer(ctx, peer)
          if scope == SCOPE_LSA:
              # Direct memory access available (NVLink)
              peer_ptr = symm_lsa_ptr(ctx, local_ptr, peer)
              data = tl.load(peer_ptr + offset)
          else:  # SCOPE_WORLD
              # Use explicit communication (GIN)
              symm_get(ctx, local_buf, remote_buf, peer, size)

        Team is obtained from the context internally.

        Args:
            ctx_ptr: Pointer to SymmContext (as int64 for Triton compatibility)
            peer: Peer rank to check (team-local rank, 0..team_size-1)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int32: SCOPE_LSA (0) if peer is in LSA domain (direct memory access),
                   SCOPE_WORLD (1) if peer requires world-scope communication,
                   -1 if context/team is invalid

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).
        """
        if backend == TL_BACKEND_DEFAULT:
            is_lsa = _symm_team_lsa_frontend(ctx_ptr, peer)
            if is_lsa == 1:
                return TL_SCOPE_LSA
            elif is_lsa == 0:
                return TL_SCOPE_WORLD
            else:
                return -1  # Error
        elif backend == TL_BACKEND_NVSHMEM:
            is_lsa = _nvshmem_symm_team_lsa(ctx_ptr, peer)
            if is_lsa == 1:
                return TL_SCOPE_LSA
            elif is_lsa == 0:
                return TL_SCOPE_WORLD
            else:
                return -1  # Error
        else:
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )
            return -1

    # =========================================================================
    # SYMM_SIGNAL - POINT-TO-POINT SIGNALING
    # =========================================================================

    @core.extern
    def _symm_signal_frontend(
        ctx_ptr,
        signal_index,
        dest_rank,
        value,
        op,
        _semantic=None,
    ):
        """
        Frontend signal operation that dispatches based on SymmContext type.

        Signal a remote rank's flag without data transfer. Atomically operates
        (set/add) on the symmetric signal at signal_index on dest_rank.
        Uses the signal pad stored in the context.
        Used for point-to-point notification.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context or invalid signal operation.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, signal_index, dest_rank, value, op],
            {
                # C function signature:
                # (int64 ctx_ptr, int32 signal_index, int32 dest_rank,
                #  int64 value, int32 op) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # signal_index
                    core.dtype("int32"),  # dest_rank
                    core.dtype("int64"),  # value
                    core.dtype("int32"),  # op
                ): ("symm_signal", core.dtype("int32")),
            },
            is_pure=False,  # Remote atomic operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_signal(
        ctx_ptr,
        signal_index,
        dest_rank,
        value,
        op,
        _semantic=None,
    ):
        """
        NVSHMEM-specific signal operation.

        Signal a remote rank's flag without data transfer using NVSHMEM atomics.
        Uses the signal pad stored in the context.
        Maps to nvshmem_uint64_atomic_set() or nvshmem_uint64_atomic_add()
        depending on the op parameter.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context or invalid signal operation.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, signal_index, dest_rank, value, op],
            {
                # C function signature:
                # (int64 ctx_ptr, int32 signal_index, int32 dest_rank,
                #  int64 value, int32 op) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # signal_index
                    core.dtype("int32"),  # dest_rank
                    core.dtype("int64"),  # value
                    core.dtype("int32"),  # op
                ): ("nvshmem_symm_signal", core.dtype("int32")),
            },
            is_pure=False,  # Remote atomic operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_SIGNAL_WAIT_UNTIL - WAIT FOR SIGNAL CONDITION
    # =========================================================================

    @core.extern
    def _symm_signal_wait_until_frontend(
        ctx_ptr,
        signal_index,
        cmp,
        cmp_value,
        _semantic=None,
    ):
        """
        Frontend signal_wait_until operation that dispatches based on SymmContext type.

        Blocks the calling thread/CTA until a local signal at signal_index meets
        the specified condition relative to the comparison value.

        Uses the gin_signal_pad from the context (same pad that symm_signal writes
        to). This enables point-to-point synchronization patterns.

        Supported conditions:
        - SIGNAL_CMP_EQ (1): Wait until signal == cmp_value
        - SIGNAL_CMP_NE (2): Wait until signal != cmp_value
        - SIGNAL_CMP_GT (3): Wait until signal > cmp_value
        - SIGNAL_CMP_GE (4): Wait until signal >= cmp_value
        - SIGNAL_CMP_LT (5): Wait until signal < cmp_value
        - SIGNAL_CMP_LE (6): Wait until signal <= cmp_value

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context or invalid comparison operation.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, signal_index, cmp, cmp_value],
            {
                # C function signature:
                # (int64 ctx_ptr, int32 signal_index, int32 cmp,
                #  int64 cmp_value) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # signal_index
                    core.dtype("int32"),  # cmp
                    core.dtype("int64"),  # cmp_value
                ): ("symm_signal_wait_until", core.dtype("int64")),
            },
            is_pure=False,  # Blocking wait operation has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_signal_wait_until(
        ctx_ptr,
        signal_index,
        cmp,
        cmp_value,
        _semantic=None,
    ):
        """
        NVSHMEM-specific signal_wait_until operation.

        Blocks the calling thread/CTA until a local signal at signal_index meets
        the specified condition relative to the comparison value.

        Uses nvshmem_signal_wait_until() from NVSHMEM.

        Supported conditions:
        - SIGNAL_CMP_EQ (1): Wait until signal == cmp_value
        - SIGNAL_CMP_NE (2): Wait until signal != cmp_value
        - SIGNAL_CMP_GT (3): Wait until signal > cmp_value
        - SIGNAL_CMP_GE (4): Wait until signal >= cmp_value
        - SIGNAL_CMP_LT (5): Wait until signal < cmp_value
        - SIGNAL_CMP_LE (6): Wait until signal <= cmp_value

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context or invalid comparison operation.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, signal_index, cmp, cmp_value],
            {
                # C function signature:
                # (int64 ctx_ptr, int32 signal_index, int32 cmp,
                #  int64 cmp_value) -> int64
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # signal_index
                    core.dtype("int32"),  # cmp
                    core.dtype("int64"),  # cmp_value
                ): ("nvshmem_symm_signal_wait_until", core.dtype("int64")),
            },
            is_pure=False,  # Blocking wait operation has side effects
            _semantic=_semantic,
        )

    # =========================================================================
    # SYMM_SIGNAL_RESET - RESET SIGNAL TO ZERO
    # =========================================================================

    @core.extern
    def _symm_signal_reset_frontend(
        ctx_ptr,
        signal_index,
        _semantic=None,
    ):
        """
        Frontend signal_reset operation that dispatches based on SymmContext type.

        Resets a local signal at signal_index to zero. This is used to prepare
        a signal for the next round of signaling/waiting in iterative algorithms.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, signal_index],
            {
                # C function signature: (int64 ctx_ptr, int32 signal_index) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # signal_index
                ): ("symm_signal_reset", core.dtype("int32")),
            },
            is_pure=False,  # Has side effects (modifies signal)
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_signal_reset(
        ctx_ptr,
        signal_index,
        _semantic=None,
    ):
        """
        NVSHMEM-specific signal_reset operation.

        Resets a local signal at signal_index to zero. This is used to prepare
        a signal for the next round of signaling/waiting in iterative algorithms.

        Uses the gin_signal_pad from the context and nvshmem_signal_wait_until
        to ensure the signal has arrived before resetting it to zero.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, signal_index],
            {
                # C function signature: (int64 ctx_ptr, int32 signal_index) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int32"),  # signal_index
                ): ("nvshmem_symm_signal_reset", core.dtype("int32")),
            },
            is_pure=False,  # Has side effects (modifies signal)
            _semantic=_semantic,
        )

    @triton.jit
    def symm_signal(
        ctx_ptr,
        signal_index,
        dest_rank,
        value: tl.constexpr = 1,
        op: tl.constexpr = 0,
        backend: tl.constexpr = 0,
    ):
        """
        Signal a remote rank's flag without data transfer.

        Atomically operates (set/add) on the symmetric signal at signal_index
        on dest_rank. Uses the signal pad stored in the context.
        Used for point-to-point notification.

        This is a non-blocking operation. To ensure the signal has been delivered,
        use symm_quiet() after the signal.

        Common usage patterns:
          # Simple notification (signal value = 1)
          symm_signal(ctx, idx, dest_rank)
          symm_quiet(ctx)  # Ensure signal is delivered

          # Signal with specific value
          symm_signal(ctx, idx, dest_rank, value=42, op=SIGNAL_OP_SET)

          # Increment a counter
          symm_signal(ctx, idx, dest_rank, value=1, op=SIGNAL_OP_ADD)

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmemx_signal_op(signal_pad + idx, value, sig_op, dest_rank)
        - NCCL: atomicExch/atomicAdd on signal_pad_ptrs[dest_rank] + idx

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            signal_index: Index into the signal buffer (int32)
            dest_rank: Destination rank/PE to signal (int32)
            value: Value to set/add (constexpr, default=1)
            op: Signal operation (constexpr, default=SIGNAL_OP_SET)
                - 0 (SIGNAL_OP_SET): Atomic set (replace value)
                - 1 (SIGNAL_OP_ADD): Atomic add (increment value)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context or invalid signal operation.
        """
        # Validate op at compile time
        tl.static_assert(
            op == TL_SIGNAL_OP_SET or op == TL_SIGNAL_OP_ADD,
            "op must be SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)",
        )

        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_signal_frontend(
                ctx_ptr,
                signal_index,
                dest_rank,
                value,
                op,
            )
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_signal(
                ctx_ptr,
                signal_index,
                dest_rank,
                value,
                op,
            )
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

    @triton.jit
    def symm_signal_wait_until(
        ctx_ptr,
        signal_index,
        cmp: tl.constexpr,
        cmp_value,
        backend: tl.constexpr = 0,
    ):
        """
        Wait until a local signal meets a specified condition.

        Blocks the calling thread/CTA until the signal at signal_index meets
        the specified condition relative to cmp_value. Uses the gin_signal_pad
        from the context (same pad that symm_signal writes to).

        This enables point-to-point synchronization patterns where one rank
        signals with symm_signal and another waits with symm_signal_wait_until.

        Common usage patterns:
          # Wait for signal to be set (equal to 1)
          symm_signal_wait_until(ctx, idx, SIGNAL_CMP_EQ, 1)

          # Wait for counter to reach threshold
          symm_signal_wait_until(ctx, idx, SIGNAL_CMP_GE, expected_count)

        Supported conditions:
        - SIGNAL_CMP_EQ (1): Wait until signal == cmp_value
        - SIGNAL_CMP_NE (2): Wait until signal != cmp_value
        - SIGNAL_CMP_GT (3): Wait until signal > cmp_value
        - SIGNAL_CMP_GE (4): Wait until signal >= cmp_value
        - SIGNAL_CMP_LT (5): Wait until signal < cmp_value
        - SIGNAL_CMP_LE (6): Wait until signal <= cmp_value

        Note: NCCL only supports SIGNAL_CMP_GE condition.

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmem_signal_wait_until(signal_pad + idx, cmp, cmp_value)
        - NCCL: ncclGin::waitSignal with ncclGin_WaitSignalGe

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            signal_index: Index into the signal buffer (int32)
            cmp: Comparison operation (constexpr)
                 - 1 (SIGNAL_CMP_EQ): Equal
                 - 2 (SIGNAL_CMP_NE): Not equal
                 - 3 (SIGNAL_CMP_GT): Greater than
                 - 4 (SIGNAL_CMP_GE): Greater than or equal
                 - 5 (SIGNAL_CMP_LT): Less than
                 - 6 (SIGNAL_CMP_LE): Less than or equal
            cmp_value: Value to compare against (int64)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                     - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                     - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                     - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Returns:
            int64: The signal value that satisfied the condition

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context or invalid comparison operation.
        """
        # Validate cmp at compile time
        tl.static_assert(
            cmp >= TL_SIGNAL_CMP_EQ and cmp <= TL_SIGNAL_CMP_LE,
            "cmp must be between SIGNAL_CMP_EQ (1) and SIGNAL_CMP_LE (6)",
        )

        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            return _symm_signal_wait_until_frontend(
                ctx_ptr,
                signal_index,
                cmp,
                cmp_value,
            )
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            return _nvshmem_symm_signal_wait_until(
                ctx_ptr,
                signal_index,
                cmp,
                cmp_value,
            )
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )
            return tl.zeros((1,), dtype=tl.int64)[0]  # Unreachable

    @triton.jit
    def symm_signal_reset(
        ctx_ptr,
        signal_index,
        backend: tl.constexpr = 0,
    ):
        """
        Reset a local signal to zero.

        Resets the signal at signal_index to zero. This is used to prepare
        a signal for the next round of signaling/waiting in iterative algorithms.

        For NVSHMEM, this uses nvshmem_signal_wait_until to ensure the signal
        has arrived before resetting it to zero, providing proper synchronization.
        For NCCL, this uses ncclGin::resetSignal to atomically reset the signal.

        Common usage patterns:
          # After waiting for a signal, reset it for next iteration
          symm_signal_wait_until(ctx, idx, SIGNAL_CMP_GE, 1)
          symm_signal_reset(ctx, idx)

          # In a loop
          for i in range(iterations):
              # ... do work ...
              symm_signal(ctx, idx, peer_rank, 1, SIGNAL_OP_SET)
              symm_signal_wait_until(ctx, idx, SIGNAL_CMP_GE, 1)
              symm_signal_reset(ctx, idx)  # Reset for next iteration

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmem_signal_wait_until + direct memory write
        - NCCL: ncclGin::resetSignal

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            signal_index: Index into the signal buffer (int32)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_signal_reset_frontend(
                ctx_ptr,
                signal_index,
            )
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_signal_reset(
                ctx_ptr,
                signal_index,
            )
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

    # =========================================================================
    # SYMM_PUT_ASYNC - NON-BLOCKING ONE-SIDED PUT
    # =========================================================================

    @core.extern
    def _symm_put_async_frontend(
        ctx_ptr,
        dest_ptr,
        src_ptr,
        count,
        element_size,
        dest_rank,
        _semantic=None,
    ):
        """
        Frontend put_async operation that dispatches based on SymmContext type.

        Non-blocking one-sided put: copies count elements from src_ptr (local)
        to dest_ptr (symmetric address) on dest_rank's buffer.

        Returns immediately without waiting for completion. Use symm_quiet to
        ensure all prior put operations have completed.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, dest_ptr, src_ptr, count, element_size, dest_rank],
            {
                # C function signature:
                # (int64 ctx_ptr, int64 dest_ptr, int64 src_ptr,
                #  int32 count, int32 element_size, int32 dest_rank) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # dest_ptr
                    core.dtype("int64"),  # src_ptr
                    core.dtype("int32"),  # count
                    core.dtype("int32"),  # element_size
                    core.dtype("int32"),  # dest_rank
                ): ("symm_put_async", core.dtype("int32")),
            },
            is_pure=False,  # Non-blocking data transfer has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_put_async(
        ctx_ptr,
        dest_ptr,
        src_ptr,
        count,
        element_size,
        dest_rank,
        _semantic=None,
    ):
        """
        NVSHMEM-specific put_async operation.

        Non-blocking one-sided put using nvshmemx_putmem_block.
        Copies count elements of element_size bytes from src_ptr to dest_ptr on dest_rank.

        Returns immediately without waiting for completion. Use symm_quiet to
        ensure all prior put operations have completed.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [ctx_ptr, dest_ptr, src_ptr, count, element_size, dest_rank],
            {
                # C function signature:
                # (int64 ctx_ptr, int64 dest_ptr, int64 src_ptr,
                #  int32 count, int32 element_size, int32 dest_rank) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # dest_ptr
                    core.dtype("int64"),  # src_ptr
                    core.dtype("int32"),  # count
                    core.dtype("int32"),  # element_size
                    core.dtype("int32"),  # dest_rank
                ): ("nvshmem_symm_put_async", core.dtype("int32")),
            },
            is_pure=False,  # Non-blocking data transfer has side effects
            _semantic=_semantic,
        )

    @triton.jit
    def symm_put_async(
        ctx_ptr,
        dest_ptr,
        src_ptr,
        count,
        dtype: tl.constexpr,
        dest_rank,
        backend: tl.constexpr = 0,
    ):
        """
        Non-blocking one-sided put operation.

        Copies count elements of the specified dtype from src_ptr (local) to
        dest_ptr (symmetric address) on dest_rank's buffer. Returns immediately
        without waiting for completion.

        This is an asynchronous operation. To ensure the data has been delivered,
        call symm_quiet() after issuing all put operations.

        Common usage patterns:
          # Simple put to a peer (float32)
          symm_put_async(ctx, remote_buf_ptr, local_data_ptr, num_elements, tl.float32, peer_rank)
          symm_quiet(ctx)  # Ensure data is delivered

          # Multiple puts before syncing (bfloat16)
          for peer in range(num_peers):
              symm_put_async(ctx, dest_ptrs[peer], src_ptr, count, tl.bfloat16, peer)
          symm_quiet(ctx)  # Ensure all puts complete

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmemx_putmem_block(dest_ptr, src_ptr, count * element_size, dest_rank)
        - NCCL: ncclGin::put with window offset resolution

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            dest_ptr: Destination pointer (symmetric address, as int64)
                      This should be a pointer to the destination in symmetric memory.
            src_ptr: Source pointer (local address, as int64)
                     This is the local data to be sent.
            count: Number of elements to transfer (int32)
            dtype: Triton data type (constexpr), e.g., tl.float32, tl.bfloat16, tl.int8
                   The element size is automatically derived from this type.
            dest_rank: Destination rank/PE number (int32)
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        # Compute element size from dtype (primitive_bitwidth is in bits)
        element_size: tl.constexpr = dtype.primitive_bitwidth // 8

        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_put_async_frontend(
                ctx_ptr,
                dest_ptr,
                src_ptr,
                count,
                element_size,
                dest_rank,
            )
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_put_async(
                ctx_ptr,
                dest_ptr,
                src_ptr,
                count,
                element_size,
                dest_rank,
            )
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

    # =========================================================================
    # SYMM_PUT_SIGNAL_ASYNC - NON-BLOCKING PUT WITH REMOTE SIGNAL
    # =========================================================================

    @core.extern
    def _symm_put_signal_async_frontend(
        ctx_ptr,
        dest_ptr,
        src_ptr,
        count,
        element_size,
        dest_rank,
        signal_index,
        signal_value,
        signal_op,
        _semantic=None,
    ):
        """
        Frontend put_signal_async operation that dispatches based on SymmContext type.

        Non-blocking one-sided put with remote signal: copies count elements from
        src_ptr (local) to dest_ptr (symmetric address) on dest_rank's buffer.
        After the data transfer completes, atomically updates the remote signal
        at signal_index on dest_rank using the specified operation.

        This is a fused operation that combines data transfer with signaling,
        allowing the receiver to know when the data has arrived without polling.
        The signal update is guaranteed to be visible only after the data transfer
        is complete.

        Returns immediately without waiting for completion. Use symm_quiet to
        ensure all prior put operations have completed.

        This calls the unified frontend function that dynamically dispatches to
        either NCCL or NVSHMEM backend based on the SymmContext type field.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [
                ctx_ptr,
                dest_ptr,
                src_ptr,
                count,
                element_size,
                dest_rank,
                signal_index,
                signal_value,
                signal_op,
            ],
            {
                # C function signature:
                # (int64 ctx_ptr, int64 dest_ptr, int64 src_ptr,
                #  int32 count, int32 element_size, int32 dest_rank,
                #  int32 signal_index, int64 signal_value, int32 signal_op) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # dest_ptr
                    core.dtype("int64"),  # src_ptr
                    core.dtype("int32"),  # count
                    core.dtype("int32"),  # element_size
                    core.dtype("int32"),  # dest_rank
                    core.dtype("int32"),  # signal_index
                    core.dtype("int64"),  # signal_value
                    core.dtype("int32"),  # signal_op
                ): ("symm_put_signal_async", core.dtype("int32")),
            },
            is_pure=False,  # Non-blocking data transfer has side effects
            _semantic=_semantic,
        )

    @core.extern
    def _nvshmem_symm_put_signal_async(
        ctx_ptr,
        dest_ptr,
        src_ptr,
        count,
        element_size,
        dest_rank,
        signal_index,
        signal_value,
        signal_op,
        _semantic=None,
    ):
        """
        NVSHMEM-specific put_signal_async operation.

        Non-blocking one-sided put with remote signal using nvshmemx_putmem_signal_block.
        Copies count elements of element_size bytes from src_ptr to dest_ptr on dest_rank.
        After the data transfer completes, atomically updates the remote signal
        at signal_index on dest_rank using the specified operation.

        Returns immediately without waiting for completion. Use symm_quiet to
        ensure all prior put operations have completed.

        This calls the NVSHMEM backend directly, bypassing runtime dispatch.
        Use this when you know the context is NVSHMEM type.

        Asserts on invalid context.
        """
        return core.extern_elementwise(
            "",  # libname - not used when extern_libs is provided
            "",  # libpath - not used when extern_libs is provided
            [
                ctx_ptr,
                dest_ptr,
                src_ptr,
                count,
                element_size,
                dest_rank,
                signal_index,
                signal_value,
                signal_op,
            ],
            {
                # C function signature:
                # (int64 ctx_ptr, int64 dest_ptr, int64 src_ptr,
                #  int32 count, int32 element_size, int32 dest_rank,
                #  int32 signal_index, int64 signal_value, int32 signal_op) -> int32
                (
                    core.dtype("int64"),  # ctx_ptr
                    core.dtype("int64"),  # dest_ptr
                    core.dtype("int64"),  # src_ptr
                    core.dtype("int32"),  # count
                    core.dtype("int32"),  # element_size
                    core.dtype("int32"),  # dest_rank
                    core.dtype("int32"),  # signal_index
                    core.dtype("int64"),  # signal_value
                    core.dtype("int32"),  # signal_op
                ): ("nvshmem_symm_put_signal_async", core.dtype("int32")),
            },
            is_pure=False,  # Non-blocking data transfer has side effects
            _semantic=_semantic,
        )

    @triton.jit
    def symm_put_signal_async(
        ctx_ptr,
        dest_ptr,
        src_ptr,
        count,
        dtype: tl.constexpr,
        dest_rank,
        signal_index,
        signal_value: tl.constexpr = 1,
        signal_op: tl.constexpr = 1,
        backend: tl.constexpr = 0,
    ):
        """
        Non-blocking one-sided put with remote signal.

        Copies count elements of the specified dtype from src_ptr (local) to
        dest_ptr (symmetric address) on dest_rank's buffer. After the data
        transfer completes, atomically updates the remote signal at signal_index
        on dest_rank using the specified operation.

        This is a fused operation that combines data transfer with signaling,
        allowing the receiver to know when the data has arrived without polling.
        The signal update is guaranteed to be visible only after the data transfer
        is complete.

        This is an asynchronous operation. To ensure the data has been delivered
        and the signal has been set, call symm_quiet() after issuing all operations.

        Common usage patterns:
          # Put data and signal completion (default: ADD 1)
          symm_put_signal_async(ctx, remote_buf, local_data, count, tl.float32, peer, sig_idx)
          # On the receiver side:
          symm_signal_wait_until(ctx, sig_idx, SIGNAL_CMP_GE, 1)

          # Put with explicit signal value and operation
          symm_put_signal_async(ctx, dest, src, count, tl.float32, peer, sig_idx,
                                signal_value=42, signal_op=SIGNAL_OP_SET)

          # Multiple puts with counting signals
          for i in range(num_chunks):
              symm_put_signal_async(ctx, dest + i*chunk_size, src + i*chunk_size,
                                    chunk_count, tl.float32, peer, sig_idx,
                                    signal_value=1, signal_op=SIGNAL_OP_ADD)
          # Receiver waits for all chunks:
          symm_signal_wait_until(ctx, sig_idx, SIGNAL_CMP_GE, num_chunks)

        This function dispatches to either the unified frontend (runtime dispatch)
        or a backend-specific implementation based on the backend hint.

        Maps to:
        - NVSHMEM: nvshmemx_putmem_signal_block(dest, src, bytes, sig_addr, value, op, pe)
        - NCCL: ncclGin::put with ncclGin_SignalAdd/ncclGin_SignalInc remote action

        Args:
            ctx_ptr: Pointer to SymmContext (NCCLSymmContext or NVSHMEMSymmContext)
            dest_ptr: Destination pointer (symmetric address, as int64)
                      This should be a pointer to the destination in symmetric memory.
            src_ptr: Source pointer (local address, as int64)
                     This is the local data to be sent.
            count: Number of elements to transfer (int32)
            dtype: Triton data type (constexpr), e.g., tl.float32, tl.bfloat16, tl.int8
                   The element size is automatically derived from this type.
            dest_rank: Destination rank/PE number (int32)
            signal_index: Index into the signal pad to update on dest_rank (int32)
            signal_value: Value to use in the signal operation (constexpr, default=1)
            signal_op: Signal operation (constexpr, default=SIGNAL_OP_ADD)
                       - 0 (SIGNAL_OP_SET): Atomic set (replace value)
                       - 1 (SIGNAL_OP_ADD): Atomic add (increment value)
                       Note: NCCL only supports SIGNAL_OP_ADD.
            backend: Backend hint (constexpr, default=0 for BACKEND_DEFAULT)
                      - 0 (BACKEND_DEFAULT): Runtime dispatch based on context type
                      - 1 (BACKEND_NCCL): Direct NCCL dispatch (not functional)
                      - 2 (BACKEND_NVSHMEM): Direct NVSHMEM dispatch

        Note:
            When using BACKEND_DEFAULT (0), use @requires_torch_symm decorator.
            When using BACKEND_NVSHMEM (2), use @requires_torch_symm(backend=BACKEND_NVSHMEM).

            This function asserts on invalid context.
        """
        # Validate signal_op at compile time
        tl.static_assert(
            signal_op == TL_SIGNAL_OP_SET or signal_op == TL_SIGNAL_OP_ADD,
            "signal_op must be SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)",
        )

        # Compute element size from dtype (primitive_bitwidth is in bits)
        element_size: tl.constexpr = dtype.primitive_bitwidth // 8

        if backend == TL_BACKEND_DEFAULT:
            # Runtime dispatch based on SymmContext type
            _symm_put_signal_async_frontend(
                ctx_ptr,
                dest_ptr,
                src_ptr,
                count,
                element_size,
                dest_rank,
                signal_index,
                signal_value,
                signal_op,
            )
        elif backend == TL_BACKEND_NVSHMEM:
            # Direct NVSHMEM dispatch
            _nvshmem_symm_put_signal_async(
                ctx_ptr,
                dest_ptr,
                src_ptr,
                count,
                element_size,
                dest_rank,
                signal_index,
                signal_value,
                signal_op,
            )
        else:
            # BACKEND_NCCL or unknown - not supported
            # NCCL does not provide device bitcode library
            tl.static_assert(
                False,
                "NCCL backend not supported (no device bitcode library available)",
            )

else:
    # Triton not available - provide stubs

    def requires_torch_symm(jit_func):  # type: ignore[misc]
        """Stub for when Triton is not available."""
        raise ImportError("Triton is required for requires_torch_symm decorator")

    def symm_all_reduce(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_all_reduce")

    def symm_quiet(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_quiet")

    def symm_barrier(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_barrier")

    def symm_barrier_arrive(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_barrier_arrive")

    def symm_barrier_wait(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_barrier_wait")

    def symm_fence(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_fence")

    def symm_lsa_ptr(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_lsa_ptr")

    def symm_lsa_multicast_ptr(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_lsa_multicast_ptr")

    def symm_team_size(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_team_size")

    def symm_team_rank(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_team_rank")

    def symm_team_peer(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_team_peer")

    def symm_signal(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_signal")

    def symm_lsa_signal_ptr(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_lsa_signal_ptr")

    def symm_signal_wait_until(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_signal_wait_until")

    def symm_signal_reset(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_signal_reset")

    def symm_put_async(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_put_async")

    def symm_put_signal_async(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Triton is required for symm_put_signal_async")


__all__ = [
    # Backend hint constants
    "BACKEND_DEFAULT",
    "BACKEND_NCCL",
    "BACKEND_NVSHMEM",
    # Reduction operation constants
    "REDUCE_OP_SUM",
    # Data type constants
    "DTYPE_FLOAT32",
    # Fence scope constants
    "FENCE_SCOPE_CTA",
    "FENCE_SCOPE_GPU",
    "FENCE_SCOPE_SYSTEM",
    # Barrier scope constants
    "SCOPE_LSA",
    "SCOPE_WORLD",
    # Signal operation constants
    "SIGNAL_OP_SET",
    "SIGNAL_OP_ADD",
    # Signal comparison condition constants
    "SIGNAL_CMP_EQ",
    "SIGNAL_CMP_NE",
    "SIGNAL_CMP_GT",
    "SIGNAL_CMP_GE",
    "SIGNAL_CMP_LT",
    "SIGNAL_CMP_LE",
    # Library finder
    "TorchSymmLibFinder",
    # Decorators
    "requires_torch_symm",
    # Triton extern functions
    "symm_all_reduce",
    # Ordering primitives
    "symm_quiet",
    "symm_barrier",
    "symm_barrier_arrive",
    "symm_barrier_wait",
    "symm_fence",
    # Memory primitives
    "symm_lsa_ptr",
    "symm_lsa_multicast_ptr",
    "symm_lsa_signal_ptr",
    # Team primitives
    "symm_team_size",
    "symm_team_rank",
    "symm_team_peer",
    # Signal primitives
    "symm_signal",
    "symm_signal_wait_until",
    "symm_signal_reset",
    # Data transfer primitives
    "symm_put_async",
    "symm_put_signal_async",
]
