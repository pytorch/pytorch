import torch

from .streams import Stream


__all__ = [
    "Event",
]


class Event:
    r"""Wrapper around an MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __init__(self, enable_timing: bool = False) -> None:
        self.__eventId = torch._C._mps_acquireEvent(enable_timing)

    def __del__(self) -> None:
        # checks if torch._C is already destroyed
        if hasattr(torch._C, "_mps_releaseEvent") and self.__eventId > 0:
            torch._C._mps_releaseEvent(self.__eventId)

    def record(self, stream: Stream | None = None) -> None:
        r"""Records the event in a given stream.

        Args:
            stream (torch.mps.Stream, torch.Stream, optional): Uses ``torch.mps.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch.mps.current_stream()
        elif stream.device.type != "mps":
            raise RuntimeError(f"expected an MPS stream, but got {stream.device.type}")
        torch._C._mps_recordEvent(self.__eventId, stream.stream_id)

    def wait(self, stream: Stream | None = None, *, value: int | None = None) -> None:
        r"""wait(stream=None, *, value=None) -> None

        Make future work submitted to the given stream wait for this event.

        Args:
            stream (torch.mps.Stream, torch.Stream, optional): Stream on which
                to wait. Default: ``None`` (the current MPS stream).

        Keyword args:
            value (int, optional): Unsigned 64-bit value that the host signal
                from :meth:`signal` must reach. A value of ``0`` is already
                satisfied. Default: ``None``, which waits for the latest
                :meth:`record`.

        Examples::

            >>> event = torch.mps.Event()  # doctest: +SKIP
            >>> event.wait(value=1)  # doctest: +SKIP
            >>> event.signal(1)  # doctest: +SKIP
        """
        if stream is None:
            stream = torch.mps.current_stream()
        elif stream.device.type != "mps":
            raise RuntimeError(f"expected an MPS stream, but got {stream.device.type}")
        if value is None:
            torch._C._mps_waitForEvent(self.__eventId, stream.stream_id)
        else:
            torch._C._mps_waitForHostSignal(self.__eventId, stream.stream_id, value)

    def signal(self, value: int) -> None:
        r"""signal(value) -> None

        Advance the host signal counter for :meth:`wait`. The counter starts
        at zero and only increases. GPU work waiting for a value at or below
        the current counter can proceed.

        A GPU wait can fail if it reaches Metal's timeout before the host signals it.

        Args:
            value (int): Unsigned 64-bit signal value.
        """
        torch._C._mps_signalEvent(self.__eventId, value)

    def query(self) -> bool:
        r"""Returns True if all work currently captured by event has completed."""
        return torch._C._mps_queryEvent(self.__eventId)

    def synchronize(self) -> None:
        r"""Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        torch._C._mps_synchronizeEvent(self.__eventId)

    def elapsed_time(self, end_event: "Event") -> float:
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return torch._C._mps_elapsedTimeOfEvents(self.__eventId, end_event.__eventId)
