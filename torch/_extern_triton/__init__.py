# Owner(s): ["oncall: distributed"]
"""
External CUDA kernel libraries for use with Triton via extern_elementwise.

This module provides CUDA kernels compiled to LLVM bitcode that can be
linked with Triton kernels using the core.extern_elementwise mechanism.

Available libraries:
- elementwise_add: Simple elementwise tensor addition operations
- symm_all_reduce: Unified symmetric memory all-reduce with NCCL/NVSHMEM dispatch

The unified symm_all_reduce interface provides a single frontend function
that automatically dispatches to either NCCL or NVSHMEM backend based on
the SymmContext type.

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

# Unified symmetric all-reduce with automatic backend dispatch
from torch._extern_triton._symm_all_reduce_triton import (
    SymmAllReduceLibFinder,
    requires_symm_all_reduce,
    symm_all_reduce_sum_f32,
)

__all__ = [
    # Elementwise add
    "requires_elementwise_add_lib",
    "scalar_add_f32",
    "scalar_add_f16",
    "scalar_add_f64",
    # Unified symmetric all-reduce
    "SymmAllReduceLibFinder",
    "requires_symm_all_reduce",
    "symm_all_reduce_sum_f32",
]
