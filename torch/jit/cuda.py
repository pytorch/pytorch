r"""
This package adds support for JIT compilation for CUDA Streams and events,
This is similar to API's available in the eager mode
:ref:`cuda-semantics` has more details about working with CUDA.
"""

import torch
from typing import Optional, Any
from torch import device as _device

def get_current_device_index() -> int:
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.

    Arguments: ``None``
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda._current_device()
    return -1

def get_device_index(device: Optional[_device] = None, optional: bool = False, allow_cpu: bool = False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    , this will return the current default CUDA device if :attr:`optional` is ``True``.
    If :attr:`allow_cpu` is ``True``,CPU devices will be accepted and ``-1`` will be
    returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if device is None:
        if optional:
            return get_current_device_index()
        else:
            raise ValueError('Expected a torch.device with a specified index '
                             f'or an integer, but got: {device}')
    device_index = -1
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(device, torch.device):
        if not allow_cpu and device.type == 'cpu':
            raise ValueError(f'Expected a non cpu device, but got: {device}')
        device_index = -1 if device.type == 'cpu' else torch.cuda.device_index(device)

    if isinstance(device, int):
        device_index = device

    return device_index

class device(object):
    r"""Context-manager that changes the selected device.
    This is similar to device (torch.device or int), but has been
    introduced for JIT compatibility.
    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: _device):
        self.idx = -1
        self.prev_idx = -1
        self.device = device

    def __enter__(self):
        self.idx = get_device_index(self.device, optional=True)

        if self.idx == -1:
            return
        self.prev_idx = torch.cuda._current_device()

        if self.prev_idx != self.idx:
            torch.cuda._set_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.prev_idx != self.idx:
            torch.cuda._set_device(self.prev_idx)

class StreamContext(object):
    r"""Context-manager that selects a given stream.
    All CUDA kernels queued within its context will be enqueued on a selected
    stream.
    Arguments:
        StreamContext (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """

    def __init__(self, stream: 'torch.classes.cuda.Stream'):
        self.idx = -1
        self.cur_stream = stream
        self.src_prev_stream = stream
        self.dst_prev_stream = stream

    def __enter__(self):
        self.idx = get_device_index(device=None, optional=True)

        if self.idx == -1:
            return
        self.src_prev_stream = torch.cuda.current_stream(self.idx)
        # If the stream is not on the current device, then change the device
        # and set the current stream on the device
        if self.src_prev_stream.device_index() != self.cur_stream.device_index():
            with device(self.cur_stream.device()):
                self.dst_prev_stream = torch.cuda.current_stream(self.cur_stream.device_index())
            torch.cuda._set_device(self.cur_stream.device_index())
        torch.cuda.set_stream(self.cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # If the stream was not on the current device, restore the previous stream on
        # the destination device and also reset the current device to the previous device.
        # Set the current stream on the device to the src_prev_stream
        if self.src_prev_stream.device_index() != self.cur_stream.device_index():
            torch.cuda.set_stream(self.dst_prev_stream)
            torch.cuda._set_device(self.idx)
        torch.cuda.set_stream(self.src_prev_stream)

def stream(stream: 'torch.classes.cuda.Stream') -> StreamContext:
    r"""Wrapper around the Context-manager that selects a given stream.
    All CUDA kernels queued within its context will be enqueued on a selected
    stream.
    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    if stream is None:
        return
    return StreamContext(stream)

def Stream(device:int = -1, priority:int = 0) -> 'torch.classes.cuda.Stream':
    r"""Wrapper around a CUDA stream.
    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`cuda-semantics` for
    details.
    Arguments:
        device(int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream. Can be either
            -1 (high priority) or 0 (low priority). By default, streams have
            priority 0.
    .. note:: Although CUDA versions >= 11 support more than two levels of
        priorities, in PyTorch, we only support two levels of priorities.
    """
    return torch.classes.cuda.Stream(device, priority)

def Event(enable_timing: bool = False,blocking: bool = False,interprocess: bool = False) -> 'torch.classes.cuda.Event':
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
