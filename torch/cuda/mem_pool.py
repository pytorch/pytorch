import contextlib

import torch
from .._utils import _dummy_type

if not hasattr(torch._C, "_CudaStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch._C.__dict__["_MemPoolContext"] = _dummy_type("_MemPoolContext")
    torch._C.__dict__["_cuda_startUsingUserPool"] = _dummy_type(
        "_cuda_startUsingUserPool"
    )
    torch._C.__dict__["_cuda_stopUsingUserPool"] = _dummy_type(
        "_cuda_stopUsingUserPool"
    )
    torch._C.__dict__["_cuda_CUDAAllocator"] = _dummy_type("_cuda_CUDAAllocator")

from torch._C import (  # noqa: F401
    _cuda_CUDAAllocator,
    _cuda_startUsingUserPool,
    _cuda_stopUsingUserPool,
    _MemPool,
    _MemPoolContext,
)


class MemPool(_MemPool):
    def __init__(self, allocator: _cuda_CUDAAllocator):
        super().__init__(allocator, True)


@contextlib.contextmanager
def use_mem_pool(pool, device=None):
    torch.cuda.init()
    ctx = _MemPoolContext(pool)
    curr_device = torch.cuda.current_device() if device is None else device
    _cuda_startUsingUserPool(curr_device)
    try:
        yield
    finally:
        _cuda_stopUsingUserPool(curr_device)
        del ctx


def empty_user_pool(pool, device=None):
    torch.cuda.init()
    curr_device = torch.cuda.current_device() if device is None else device
    torch._C._cuda_emptyUserPool(curr_device, pool)
