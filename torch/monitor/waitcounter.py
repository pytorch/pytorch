import os

from torch._C._monitor import _WaitCounter as _OriginalWaitCounter, _WaitCounterTracker


class _NoopWaitCounterTracker:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object, **kwargs: object) -> None:
        pass


class _NoopWaitCounter:
    def __init__(self, key: str) -> None:
        pass

    def guard(self) -> _WaitCounterTracker:
        return _NoopWaitCounterTracker()  # type: ignore[return-value]


class _WaitCounter:
    """
    Wrapper around torch._C._monitor._WaitCounter to control enable/disable
    """

    def __new__(cls, key: str) -> _OriginalWaitCounter:  # type: ignore[misc]
        if os.getenv("TORCH_DISABLE_WAIT_COUNTERS", "0") == "1":
            return _NoopWaitCounter(key)  # type: ignore[return-value]
        else:
            return _OriginalWaitCounter(key)
