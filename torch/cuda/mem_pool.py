import contextlib

from typing import Any, Dict, Optional, Union

import torch
from .._utils import _dummy_type
from . import _get_device_index


if not hasattr(torch._C, "_CudaStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch._C.__dict__["_MemPoolContext"] = _dummy_type("_MemPoolContext")
    torch._C.__dict__["_cuda_startUsingUserPool"] = _dummy_type(
        "_cuda_startUsingUserPool"
    )
    torch._C.__dict__["_cuda_releasePool"] = _dummy_type("_cuda_releasePool")
    torch._C.__dict__["_cuda_getPoolUseCount"] = _dummy_type("_cuda_getPoolUseCount")
    torch._C.__dict__["_cuda_CUDAAllocator"] = _dummy_type("_cuda_CUDAAllocator")

from torch._C import (  # noqa: F401
    _cuda_CUDAAllocator,
    _cuda_endAllocateCurrentStreamToPool,
    _cuda_getPoolUseCount,
    _cuda_releasePool,
    _cuda_startUsingUserPool,
    _MemPool,
    _MemPoolContext,
)


class MemPool(_MemPool):
    def __init__(self, allocator: Optional[_cuda_CUDAAllocator] = None):
        super().__init__(allocator, True)

    def use_count(self, device: Union[Device, int] = None):
        torch.cuda.init()
        device_index = (
            torch.cuda.current_device() if device is None else _get_device_index(device)
        )
        return _cuda_getPoolUseCount(device_index, self.id)

    def release(self, device: Union[Device, int] = None):
        torch.cuda.init()
        device_index = (
            torch.cuda.current_device() if device is None else _get_device_index(device)
        )
        for _ in range(self.use_count()):
            _cuda_releasePool(device_index, self.id)

    def empty_cache(self, device: Union[Device, int] = None):
        torch.cuda.init()
        ctx = _MemPoolContext(self)
        torch.cuda.empty_cache(device, self.id)
        del ctx

    def snapshot(self, device: Union[Device, int] = None):
        torch.cuda.init()
        return torch.cuda.memory_snapshot(device, self.id)


@contextlib.contextmanager
def use_mem_pool(pool, device: Union[Device, int] = None):
    torch.cuda.init()
    ctx = _MemPoolContext(pool)
    device_index = (
        torch.cuda.current_device() if device is None else _get_device_index(device)
    )
    _cuda_startUsingUserPool(device_index)
    try:
        yield
    finally:
        _cuda_endAllocateCurrentStreamToPool(device_index, pool.id)
        del ctx
