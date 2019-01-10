import ctypes
import torch
from . import cudart, check_error, cudaStatus
from ._utils import _get_device_index
from torch._C import _add_docstr


class Stream(torch._C._CudaStreamBase):
    r"""Wrapper around a CUDA stream.

    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`cuda-semantics` for
    details.

    Arguments:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream. Lower numbers
                                 represent higher priorities.
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        with torch.cuda.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Arguments:
            event (Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see `CUDA
           documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA documentation:
           http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Arguments:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Arguments:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed."""
        return super(Stream, self).query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cudaStreamSynchronize()``: see
           `CUDA documentation`_ for more info.

        .. _CUDA documentation:
           http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        super(Stream, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        return ('<torch.cuda.Stream device={0} cuda_stream={1:#x}>'
                .format(self.device, self.cuda_stream))


class EventHandle(ctypes.Structure):
    IPC_HANDLE_SIZE = 64
    _fields_ = [('reserved', ctypes.c_char * IPC_HANDLE_SIZE)]


class Event(torch._C._CudaEventBase):
    r"""Wrapper around CUDA event.

    Arguments:
        enable_timing (bool): indicates if the event should measure time
            (default: ``False``)
        blocking (bool): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False,
                **kwargs):
        return super(Event, cls).__new__(
            cls, enable_timing=enable_timing, blocking=blocking,
            interprocess=interprocess, **kwargs)

    def record(self, stream=None):
        r"""Records the event in a given stream."""
        if stream is None:
            stream = torch.cuda.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes a given stream wait for the event."""
        if stream is None:
            stream = torch.cuda.current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if the event has been recorded.

        Returns:
            A boolean indicating if the event has been recorded.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds before the event was recorded."""
        return super(Event, self).elapsed_time(end_event)

    def synchronize(self):
        r"""Synchronizes with the event."""
        super(Event, self).synchronize()

    def ipc_handle(self):
        r"""Returns an IPC handle of this event."""
        handle = EventHandle()
        return super(Event, self).ipc_handle(ctypes.byref(handle))
        return handle

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_event)

    def __repr__(self):
        return '<torch.cuda.Event {0:#x}>'.format(self._as_parameter_.value)
