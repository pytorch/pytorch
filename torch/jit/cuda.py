# mypy: ignore-errors

r"""
This package adds support for JIT compilation for CUDA Streams and events,
This is similar to API's available in the eager mode
:ref:`cuda-semantics` has more details about working with CUDA.
"""

import torch
from typing import Optional, Any
from torch._utils import _get_device_index

def get_current_device_index() -> int:
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.

    Arguments: ``None``
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    return -1

class device(object):
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return

        self.prev_idx = torch.cuda.current_device()
        if self.prev_idx != self.idx:
            torch.cuda.set_device(self.idx)
            # Call lazy initialization if we are in eager mode.
            if not torch.jit.is_scripting():
                _lazy_init()

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.prev_idx != self.idx:
            torch.cuda.set_device(self.prev_idx)
        return False

class StreamContext(object):
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    cur_stream : Optional['torch.classes.cuda.Stream']

    def __init__(self, stream: Optional['torch.classes.cuda.Stream']):  # type: ignore
        self.idx = -1
        self.stream = stream
        # Initialize the below streams to default stream on the current device
        self.device_index = _get_device_index(device=None, optional=True)
        self.src_prev_stream = torch.cuda.default_stream(self.device_index)
        self.dst_prev_stream = torch.cuda.default_stream(self.device_index)

    def __enter__(self):
        self.idx = _get_device_index(device=None, optional=True)
        # If there is no CUDA device available, return
        if self.idx == -1:
            return

        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None
        if cur_stream is None:
            return
        self.src_prev_stream = torch.cuda.current_stream(self.idx)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                cur_stream_device_index = _get_device_index(cur_stream.device)
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream_device_index)
        torch.cuda.set_stream(cur_stream)  # type: ignore

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:
            torch.cuda.set_stream(self.dst_prev_stream)  # type: ignore
        torch.cuda.set_stream(self.src_prev_stream)  # type: ignore

def stream(stream: Optional['torch.classes.cuda.Stream']) -> StreamContext:  # type: ignore
    r"""Wrapper around the Context-manager StreamContextthat
        selects a given stream.
    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    ..Note:: In eager mode stream is of type Stream class while in JIT it is
    an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)

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
