import ctypes
import contextlib
from typing import Optional

import torch
from ._utils import _dummy_type

if not hasattr(torch._C, "_CudaStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch._C.__dict__["_MemPoolContext"] = _dummy_type("_MemPoolContext")
    torch._C.__dict__["_cuda_startUsingUserPool"] = _dummy_type("_cuda_startUsingUserPool")
    torch._C.__dict__["_cuda_stopUsingUserPool"] = _dummy_type("_cuda_stopUsingUserPool")

from torch._C import (  # noqa: F401
    _MemPool,
    _MemPoolContext,
    _cuda_startUsingUserPool,
    _cuda_stopUsingUserPool,
)

class MemPool(_MemPool):
    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        return super().__init__(alloc_fn, free_fn, True)


@contextlib.contextmanager
def use_mem_pool(pool, device=None):
    torch.cuda.init()
    ctx = _MemPoolContext(pool)
    curr_device = torch.cuda.current_device() if device == None else device
    _cuda_startUsingUserPool(curr_device)
    try:
        yield
    finally:
        _cuda_stopUsingUserPool(curr_device)
        del ctx


def empty_user_pool(pool, device=None):
    torch.cuda.init()
    curr_device = torch.cuda.current_device() if device == None else device
    torch._C._cuda_emptyUserPool(curr_device, pool)
