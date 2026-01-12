# mypy: allow-untyped-defs
"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""

import multiprocessing
import sys

import torch
from .reductions import init_reductions


__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__  # noqa: PLE0605 type: ignore[attr-defined]


# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
torch._C._multiprocessing_init()


"""Add helper function to spawn N processes and wait for completion of any of
them."""
from .spawn import (
    ENV_VAR_PARALLEL_START,
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
    SpawnContext,
    start_processes,
)


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    if new_strategy not in _all_sharing_strategies:
        raise AssertionError(
            f"invalid sharing strategy {new_strategy!r}, "
            f"expected one of {_all_sharing_strategies}"
        )
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


def _set_thread_name(name: str) -> None:
    """Set the name of the current thread.

    Args:
        name (str): Name of the current thread.
    """
    torch._C._set_thread_name(name)


def _get_thread_name() -> str:
    """Get the name of the current thread.

    Returns:
        str: Name of the current thread.
    """
    return torch._C._get_thread_name()


init_reductions()

# Leak ResourceTracker at exit for Python-3.12 on MacOS
# See https://github.com/pytorch/pytorch/issues/153050 and
# https://github.com/python/cpython/issues/88887 for more details
from multiprocessing.resource_tracker import ResourceTracker as _RT


if (
    sys.platform == "darwin"
    and sys.version_info >= (3, 12, 2)
    and hasattr(_RT, "__del__")
):
    import atexit

    def _leak_RT_at_exit():
        def _noop(x):
            pass

        _RT.__del__ = _noop  # type: ignore[attr-defined]

    atexit.register(_leak_RT_at_exit)
