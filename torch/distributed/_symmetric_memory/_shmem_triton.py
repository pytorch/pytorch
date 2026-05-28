"""
SHMEM Triton backend dispatch.

Thin abstraction layer that routes to the correct platform-specific module:
  - ROCm  - _rocshmem_triton
  - CUDA  - _nvshmem_triton


End users import from here to get backend-agnostic functionality:
  - get_shmem_backend_module(): returns the active backend module
  - requires_shmem(): backend-agnostic @requires_shmem decorator
"""

import torch


def get_shmem_backend_module():
    if torch.version.hip is not None:
        from torch.distributed._symmetric_memory import _rocshmem_triton

        return _rocshmem_triton
    from torch.distributed._symmetric_memory import _nvshmem_triton

    return _nvshmem_triton


def requires_shmem(jit_func):  # type: ignore[no-untyped-def]
    """
    Backend-agnostic Triton decorator for SHMEM kernels.

    Delegates to ``@requires_rocshmem`` on ROCm and ``@requires_nvshmem``
    on CUDA.
    """
    backend = get_shmem_backend_module()
    if torch.version.hip is not None:
        return backend.requires_rocshmem(jit_func)
    return backend.requires_nvshmem(jit_func)
