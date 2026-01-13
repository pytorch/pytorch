# Owner(s): ["oncall: distributed"]
"""
External CUDA kernel libraries for use with Triton via extern_elementwise.

This module provides CUDA kernels compiled to LLVM bitcode that can be
linked with Triton kernels using the core.extern_elementwise mechanism.

Available libraries:
- elementwise_add: Simple elementwise tensor addition operations
- symm_all_reduce: Symmetric memory all-reduce operations using NCCL
"""

from torch._extern_triton._elementwise_add_triton import (
    requires_elementwise_add_lib,
    scalar_add_f16,
    scalar_add_f32,
    scalar_add_f64,
)
from torch._extern_triton._symm_all_reduce_triton import (
    requires_symm_all_reduce_lib,
    symm_all_reduce_sum_f32,
)


__all__ = [
    # Elementwise add
    "requires_elementwise_add_lib",
    "scalar_add_f32",
    "scalar_add_f16",
    "scalar_add_f64",
    # Symmetric all-reduce
    "requires_symm_all_reduce_lib",
    "symm_all_reduce_sum_f32",
]
