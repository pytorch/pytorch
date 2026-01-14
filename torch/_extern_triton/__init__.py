# Owner(s): ["oncall: distributed"]
"""
External CUDA kernel libraries for use with Triton via extern_elementwise.

This module provides CUDA kernels compiled to LLVM bitcode that can be
linked with Triton kernels using the core.extern_elementwise mechanism.

Available libraries:
- elementwise_add: Simple elementwise tensor addition operations
- torch_symm: Unified symmetric memory primitives with NCCL/NVSHMEM dispatch

The unified torch_symm interface provides frontend functions that
automatically dispatch to either NCCL or NVSHMEM backend based on
the SymmContext type.

Backend hint support:
The symm_all_reduce_sum_f32 function accepts a constexpr backend argument:
- BACKEND_DEFAULT (0): Runtime dispatch based on context type (requires torch_symm.bc)
- BACKEND_NCCL (1): Direct NCCL dispatch (not functional - no device bitcode)
- BACKEND_NVSHMEM (2): Direct NVSHMEM dispatch (only needs libnvshmem_device.bc)

Backend support:
- NVSHMEM: Fully functional (provides libnvshmem_device.bc)
- NCCL: NOT functional (NCCL does not provide device bitcode library)
"""

from torch._extern_triton._elementwise_add_triton import (
    requires_elementwise_add_lib,
    scalar_add_f16,
    scalar_add_f32,
    scalar_add_f64,
)

# Unified symmetric primitives with automatic backend dispatch
from torch._extern_triton._torch_symm_triton import (
    BACKEND_DEFAULT,
    BACKEND_NCCL,
    BACKEND_NVSHMEM,
    SymmAllReduceLibFinder,
    TorchSymmLibFinder,
    requires_symm_all_reduce,
    requires_torch_symm,
    symm_all_reduce_sum_f32,
)

__all__ = [
    # Elementwise add
    "requires_elementwise_add_lib",
    "scalar_add_f32",
    "scalar_add_f16",
    "scalar_add_f64",
    # Backend hint constants
    "BACKEND_DEFAULT",
    "BACKEND_NCCL",
    "BACKEND_NVSHMEM",
    # Torch symmetric memory
    "TorchSymmLibFinder",
    "SymmAllReduceLibFinder",  # Backward compatibility alias
    "requires_torch_symm",
    "requires_symm_all_reduce",  # Backward compatibility alias
    "symm_all_reduce_sum_f32",
]
