import torch


class Event:
    r"""Wrapper around an MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __init__(self, enable_timing=False):
        self.__eventId = torch._C._mps_acquireEvent(enable_timing)

    def __del__(self):
        # checks if torch._C is already destroyed
        if hasattr(torch._C, "_mps_releaseEvent") and self.__eventId > 0:
            torch._C._mps_releaseEvent(self.__eventId)

    def record(self):
        r"""Records the event in the default stream."""
        torch._C._mps_recordEvent(self.__eventId)

    def wait(self):
        r"""Makes all future work submitted to the default stream wait for this event."""
        torch._C._mps_waitForEvent(self.__eventId)

    def query(self):
        r"""Returns True if all work currently captured by event has completed."""
        return torch._C._mps_queryEvent(self.__eventId)

    def synchronize(self):
        r"""Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        torch._C._mps_synchronizeEvent(self.__eventId)

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return torch._C._mps_elapsedTimeOfEvents(self.__eventId, end_event.__eventId)
