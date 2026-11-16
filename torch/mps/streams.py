from __future__ import annotations

from contextlib import contextmanager

import torch
from torch._utils import _dummy_type

from .event import Event


# APIs defined in this file are public directly under `torch.mps`
__all__: list[str] = []

if not hasattr(torch._C, "_MPSStreamBase"):
    torch._C.__dict__["_MPSStreamBase"] = _dummy_type("_MPSStreamBase")


class Stream(torch._C._MPSStreamBase):
    r"""Wrapper around an MPS stream.

    An MPS stream is a linear sequence of execution that is independent from
    other streams. Use the :meth:`stream` context manager to ensure operators
    run on the corresponding stream.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
    """

    def __new__(cls, device=None, **kwargs):
        if not torch.backends.mps.is_built():
            raise RuntimeError("torch.mps.Stream requires MPS support")

        # Currently all Apple M-series machines have only one GPU. So if
        # `device` is given, just check it for errors but don't actually use it.
        if device is not None:
            if isinstance(device, int):
                if device > 0:
                    raise ValueError(f"mps device index is invalid: {device}")
            else:
                device = torch.device(device)
                if device.type != "mps":
                    raise ValueError(f"Expected an mps device, but got: {device.type}")
                if device.index is not None and device.index != 0:
                    raise ValueError(f"mps device index is invalid: {device.index}")

        return super().__new__(cls, **kwargs)

    def wait_event(self, event: Event | torch.Event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (Event, torch.Event): an event to wait for.

        .. note:: This function returns without waiting for :attr:`event`:. Only
           future operations are affected.
        """
        event.wait(self)

    def wait_stream(self, stream: Stream | torch.Stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream, torch.Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels. Only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event: Event | torch.Event | None = None):
        r"""Record an event.

        Args:
            event (Event, torch.Event, optional): event to record. If not given,
                a new one will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    @property
    def stream_id(self) -> int:
        return super().stream_id

    def __repr__(self):
        return f"<torch.mps.Stream stream_id={self.stream_id}>"


@contextmanager
def stream(stream: Stream | None):
    r"""Context manager that selects a given stream.

    MPS operators called within this context will be enqueued on the selected
    stream.

    Args:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    prev_stream = torch.mps.current_stream()

    if stream is not None:
        torch.mps.set_stream(stream)

    try:
        yield
    finally:
        if stream is not None:
            torch.mps.set_stream(prev_stream)


def current_stream() -> Stream:
    r"""Return the currently selected :class:`Stream`."""
    stream_base = torch._C._mps_getCurrentStream()
    return Stream(stream_id=stream_base.stream_id)


def default_stream() -> Stream:
    r"""Return the default :class:`Stream`."""
    return Stream(stream_id=0)


def set_stream(stream: Stream | None) -> None:
    r"""Set the current stream. Usage of this function is discouraged in favor
    of the :meth:`stream` context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op if this
            argument is ``None``.
    """
    if stream is None:
        return
    torch._C._mps_setStream(stream)


for _obj in (Stream, stream, current_stream, default_stream, set_stream):
    _obj.__module__ = "torch.mps"
del _obj
