from typing import Optional

import torch
from .._utils import _dummy_type
from .memory import CUDAPluggableAllocator

if not hasattr(torch._C, "_MemPool"):
    # Define dummy base classes
    torch._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch._C.__dict__["_MemPoolContext"] = _dummy_type("_MemPoolContext")

from torch._C import _MemPool, _MemPoolContext  # noqa: F401


class MemPool(_MemPool):
    r"""MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the CUDACachingAllocator.

    Args:
        allocator(torch.cuda.memory.CUDAPluggableAllocator, optional): a
            CUDAPluggableAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the CUDACachingAllocator.

    """

    def __init__(self, allocator: Optional[CUDAPluggableAllocator] = None):
        super().__init__(None if allocator is None else allocator.allocator(), True)

    @property
    def id(self):
        r"""Returns the ID of this pool as a tuple of two ints."""
        return super().id

    @property
    def allocator(self):
        r"""Returns the allocator this MemPool routes allocations to"""
        return super().allocator


class MemPoolContext(_MemPoolContext):
    r"""MemPoolContext holds the currently active pool and stashes the previous
    pool. On deletion it makes the previous pool active.

    Args:
        pool(torch.cuda.MemPool): a MemPool object to be made active so that
        allocations route to this pool.

    """

    def __init__(self, pool: MemPool):
        super().__init__(pool)

    @staticmethod
    def active_pool():
        r"""Returns the active MemPool"""
        return _MemPoolContext.active_pool()
