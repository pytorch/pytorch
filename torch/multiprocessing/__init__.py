import sys
from .reductions import init_reductions
import multiprocessing

__all__ = ['set_sharing_strategy', 'get_sharing_strategy',
           'get_all_sharing_strategies']


from multiprocessing import *


__all__ += multiprocessing.__all__


if sys.version_info < (3, 3):
    """Override basic classes in Python 2.7 and Python 3.3 to use ForkingPickler
    for serialization. Later versions of Python already use ForkingPickler."""
    from .queue import Queue, SimpleQueue
    from .pool import Pool


if sys.platform == 'darwin':
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


init_reductions()
