from filelock import FileLock as base_FileLock
from torch.monitor import _WaitCounter

from typing import Self
from types import TracebackType


class WaitCounterFileLock(base_FileLock):
    def __enter__(self) -> Self:
        self.counter = _WaitCounter("pytorch.filelock").guard()
        self.counter.__enter__()
        super().__enter__()
        return self

    def __exit__(
            self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Self:
        super().__exit__(exc_type, exc_value, traceback)
        self.counter.__exit__()
