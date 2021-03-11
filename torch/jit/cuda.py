# mypy: ignore-errors

r"""
This package adds support for JIT compilation for CUDA Streams and events,
This is similar to API's available in the eager mode
:ref:`cuda-semantics` has more details about working with CUDA.
"""

import torch
from typing import Optional

def get_current_device_index() -> int:
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.

    Arguments: ``None``
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    return -1

def Stream(device: Optional[torch.device] = None, priority: int = 0) -> 'torch.classes.cuda.Stream':
    r"""Wrapper around a CUDA stream.
    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`cuda-semantics` for
    details.
    Arguments:
        device(torch.device, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default), stream will be
            created on the current device.
        priority(int, optional): priority of the stream. Can be either
            -1 (high priority) or 0 (low priority). By default, streams have
            priority 0.
    .. note:: Although CUDA versions >= 11 support more than two levels of
        priorities, in PyTorch, we only support two levels of priorities.
    """
    return torch.classes.cuda.Stream(device, priority)

def Event(enable_timing: bool = False, blocking: bool = False, interprocess: bool = False) -> 'torch.classes.cuda.Event':
    r"""Wrapper around a CUDA event.
    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.
    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
    .. _CUDA Event Documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    """
    return torch.classes.cuda.Event(enable_timing, blocking, interprocess)
