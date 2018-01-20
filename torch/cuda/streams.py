import ctypes
import torch
from . import cudart, check_error, cudaStatus


class Stream(torch._C._CudaStreamBase):
    """Wrapper around a CUDA stream.

    A `CUDA stream`_ is an independent sequence of execution that belongs to a
    specific device.  If you do not create a stream explicitly, each device uses
    its own "default" stream.

    Operations inside each stream is serialized in the order they are created,
    but operations from different streams can execute concurrently in any
    relative order, unless explicit synchronization instructions (such as
    :meth:`~synchronize` or :meth:`~wait_stream`) are used.  For example, the
    following code is incorrect:

        >>> s = torch.cuda.stream()  # Create a new stream.
        >>> A = torch.cuda.FloatTensor(100, 100).normal_(0.0, 1.0)
        >>> with torch.cuda.stream(s):
        >>>     # sum() may start execution before normal_() finishes!
        >>>     B = torch.sum(A)

    As convenience, when the "current stream" is the default stream, PyTorch
    automatically performs necessary synchronization when copying data between
    CPU and GPU or between two GPUs.  Hence, as long as you do not explicitly
    create a (non-default) stream (or explicitly request asynchronous operation
    in, e.g., :meth:`~torch.Tensor.copy_`), your code will run as if every
    operation was executed synchronously.  However, when using non-default
    streams, it is the user's responsibility to ensure proper synchronization.

    Arguments:
        device(int, optional): a device on which to allocate the Stream.
        priority(int, optional): priority of the stream. Lower numbers
                                 represent higher priorities.

    .. _CUDA stream: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
    """

    def __new__(cls, device=-1, priority=0, **kwargs):
        with torch.cuda.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        """Makes all future work submitted to the stream wait for an event.

        Arguments:
            event (Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see `CUDA
           documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA documentation:
           http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        check_error(cudart().cudaStreamWaitEvent(self, event, ctypes.c_int(0)))

    def wait_stream(self, stream):
        """Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Arguments:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        """Records an event.

        Arguments:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        check_error(cudart().cudaEventRecord(event, self))
        return event

    def query(self):
        """Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        res = cudart().cudaStreamQuery(self)
        if res == cudaStatus.ERROR_NOT_READY:
            return False
        check_error(res)
        return True

    def synchronize(self):
        """Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cudaStreamSynchronize()``: see
           `CUDA documentation`_ for more info.

        .. _CUDA documentation:
           http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        check_error(cudart().cudaStreamSynchronize(self))

    @staticmethod
    def priority_range():
        least_priority = ctypes.c_int()
        greatest_priority = ctypes.c_int()
        check_error(cudart().cudaDeviceGetStreamPriorityRange(
            ctypes.byref(least_priority), ctypes.byref(greatest_priority)))
        return (least_priority.value, greatest_priority.value)

    @property
    def priority(self):
        priority = ctypes.c_int()
        check_error(cudart().cudaStreamGetPriority(self, ctypes.byref(priority)))
        return priority.value

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return o.device == self.device and o.cuda_stream == self.cuda_stream
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        return ('<torch.cuda.Stream device={0} cuda_stream={1:#x}>'
                .format(self.device, self.cuda_stream))


class EventHandle(ctypes.Structure):
    IPC_HANDLE_SIZE = 64
    _fields_ = [('reserved', ctypes.c_char * IPC_HANDLE_SIZE)]


class Event(object):
    """Wrapper around CUDA event.

    Arguments:
        enable_timing (bool): indicates if the event should measure time
            (default: ``False``)
        blocking (bool): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
    """

    DEFAULT = 0x0
    BLOCKING_SYNC = 0x1
    DISABLE_TIMING = 0x2
    INTERPROCESS = 0x4

    def __init__(self, enable_timing=False, blocking=False, interprocess=False,
                 _handle=None):
        flags = Event.DEFAULT
        if not enable_timing:
            flags |= Event.DISABLE_TIMING
        if blocking:
            flags |= Event.BLOCKING_SYNC
        if interprocess:
            flags |= Event.INTERPROCESS

        ptr = ctypes.c_void_p()
        self._cudart = cudart()
        if _handle:
            check_error(self._cudart.cudaIpcOpenEventHandle(ctypes.byref(ptr), _handle))
        else:
            check_error(self._cudart.cudaEventCreateWithFlags(ctypes.byref(ptr), ctypes.c_uint(flags)))
        self._as_parameter_ = ptr

    def __del__(self):
        if hasattr(self, '_as_parameter_'):
            check_error(self._cudart.cudaEventDestroy(self._as_parameter_))
            del self._as_parameter_

    def record(self, stream=None):
        """Records the event in a given stream."""
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.record_event(self)

    def wait(self, stream=None):
        """Makes a given stream wait for the event."""
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.wait_event(self)

    def query(self):
        """Checks if the event has been recorded.

        Returns:
            A boolean indicating if the event has been recorded.
        """
        res = cudart().cudaEventQuery(self)
        if res == cudaStatus.ERROR_NOT_READY:
            return False
        check_error(res)
        return True

    def elapsed_time(self, end_event):
        """Returns the time elapsed before the event was recorded."""
        time_ms = ctypes.c_float()
        check_error(cudart().cudaEventElapsedTime(
            ctypes.byref(time_ms), self, end_event))
        return time_ms.value

    def synchronize(self):
        """Synchronizes with the event."""
        check_error(cudart().cudaEventSynchronize(self))

    def ipc_handle(self):
        """Returns an IPC handle of this event."""
        handle = EventHandle()
        check_error(cudart().cudaIpcGetEventHandle(ctypes.byref(handle), self))
        return handle

    def __repr__(self):
        return '<torch.cuda.Event {0:#x}>'.format(self._as_parameter_.value)
