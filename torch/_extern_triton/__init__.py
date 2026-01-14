# Owner(s): ["oncall: distributed"]
"""
External CUDA kernel libraries for use with Triton via extern_elementwise.

This module provides CUDA kernels compiled to LLVM bitcode that can be
linked with Triton kernels using the core.extern_elementwise mechanism.

Available libraries:
- torch_symm: Unified symmetric memory primitives with NCCL/NVSHMEM dispatch

The torch_symm interface provides frontend functions that automatically
dispatch to either NCCL or NVSHMEM backend based on the SymmContext type.

Backend hint support:
The symm_all_reduce_sum_f32 function accepts a constexpr backend argument:
- BACKEND_DEFAULT (0): Runtime dispatch based on context type (requires torch_symm.bc)
- BACKEND_NCCL (1): Direct NCCL dispatch (not functional - no device bitcode)
- BACKEND_NVSHMEM (2): Direct NVSHMEM dispatch (only needs libnvshmem_device.bc)

Backend support:
- NVSHMEM: Fully functional (provides libnvshmem_device.bc)
- NCCL: NOT functional (NCCL does not provide device bitcode library)
"""

# Unified symmetric primitives with automatic backend dispatch
from torch._extern_triton._torch_symm_triton import (
    BACKEND_DEFAULT,
    BACKEND_NCCL,
    BACKEND_NVSHMEM,
    TorchSymmLibFinder,
    requires_torch_symm,
    symm_all_reduce_sum_f32,
)

__all__ = [
    # Backend hint constants
    "BACKEND_DEFAULT",
    "BACKEND_NCCL",
    "BACKEND_NVSHMEM",
    # Torch symmetric memory
    "TorchSymmLibFinder",
    "requires_torch_symm",
    "symm_all_reduce_sum_f32",
]
