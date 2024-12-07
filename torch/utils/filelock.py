from types import TracebackType
from typing import Self

from filelock import FileLock as base_FileLock

from torch.monitor import _WaitCounter


class FileLock(base_FileLock):
    """
    This behaves like a normal file lock.

    However, it adds waitcounters for acquiring and releasing the filelock
    as well as for the critical region within it.

    pytorch.filelock.enter - While we're acquiring the filelock.
    pytorch.filelock.region - While we're holding the filelock and doing work.
    pytorch.filelock.exit - While we're releasing the filelock.
    """

    def __enter__(self) -> Self:
        self.region_counter = _WaitCounter("pytorch.filelock.region").guard()
        self.region_counter.__enter__()
        with _WaitCounter("pytorch.filelock.enter").guard():
            super().__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with _WaitCounter("pytorch.filelock.exit").guard():
            super().__exit__(exc_type, exc_value, traceback)
        self.region_counter.__exit__()
        return None
