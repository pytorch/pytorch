from sys import platform as _platform
from multiprocessing import *


if _platform == 'darwin':
    _sharing_strategy = 'file_system'
    _all_sharing_strategies = {'file_system'}
else:
    _sharing_strategy = 'file_descriptor'
    _all_sharing_strategies = {'file_descriptor', 'file_system'}


def set_sharing_strategy(new_strategy):
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    return _sharing_strategy

def get_all_sharing_strategies():
    return _all_sharing_strategies


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
