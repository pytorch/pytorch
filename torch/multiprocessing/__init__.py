from multiprocessing import *


_sharing_strategy = 'file_descriptor'


def set_sharing_strategy(new_stragegy):
    global _sharing_strategy
    assert new_stragegy in {'file_descriptor', 'file_system'}
    _sharing_strategy = new_stragegy


def get_sharing_strategy():
    return _sharing_strategy


def Queue(*args, **kwargs):
    from .queue import Queue, FdQueue
    if _sharing_strategy == 'file_descriptor':
        return FdQueue(*args, **kwargs)
    elif _sharing_strategy == 'file_system':
        return Queue(*args, **kwargs)


from .pool import Pool
from ._storage import _init_storage_sharing
from ._tensor import _init_tensor_sharing
_init_storage_sharing()
_init_tensor_sharing()

