# mypy: allow-untyped-defs
import ctypes

import torch
from torch._utils import _dummy_type


if not hasattr(torch._C, "_CudaStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_CudaStreamBase"] = _dummy_type("_CudaStreamBase")
    torch._C.__dict__["_CudaEventBase"] = _dummy_type("_CudaEventBase")


class Stream(torch._C._CudaStreamBase):
    r"""Wrapper around a CUDA stream.

    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.  See :ref:`cuda-semantics` for details.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.
            If the value falls outside of the allowed priority range, it will automatically be
            mapped to the nearest valid priority (lowest for large positive numbers or
            highest for large negative numbers).

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.cuda.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.cuda.Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
           `CUDA Stream documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA Stream documentation:
           https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        event.wait(self)

    def wait_stream(self, stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Record an event.

        Args:
            event (torch.cuda.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cudaStreamSynchronize()``: see
           `CUDA Stream documentation`_ for more info.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        return f"<torch.cuda.Stream device={self.device} cuda_stream={self.cuda_stream:#x}>"

    def __cuda_stream__(self):
        """Implements the CUDA Stream Protocol:
        https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol

        Returns:
            tuple: A 2-tuple of (version, handle) where version is the protocol version
                   and handle is the address of cudaStream_t (CUDA) or hipStream_t (ROCm) as a Python int.
        """
        return (0, self.cuda_stream)


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated CUDA stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `cudaStream_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch.cuda.device(device):
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)


class Event(torch._C._CudaEventBase):
    r"""Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    The underlying CUDA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
        external (bool, optional): indicates whether this event should create event record and event wait nodes, or create an internal cross-stream dependency, when captured in a cuda graph. See `cross-stream dependencies <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events>`_, `cudaEventRecordExternal <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3457b81d1d32c6a00f6132fbc2693d47>`_, and `cudaEventWaitExternal <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0c23426b7252eaa9cef695859991304e>`_ for more information about internal vs. external events. (default: ``False``)

    .. _CUDA Event Documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    """  # noqa: B950

    def __new__(
        cls, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
            external=external,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super().from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Record the event in a given stream.

        Uses ``torch.cuda.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        super().record(stream)

    def wait(self, stream=None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch.cuda.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
            `CUDA Event documentation`_ for more info.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        super().wait(stream)

    def query(self):
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cudaEventSynchronize()``: see
            `CUDA Event documentation`_ for more info.
        """
        super().synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_event)

    def __repr__(self) -> str:
        if self.cuda_event:
            return f"<torch.cuda.Event {self._as_parameter_.value:#x}>"
        else:
            return "<torch.cuda.Event uninitialized>"
