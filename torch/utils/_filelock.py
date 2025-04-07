from types import TracebackType
from typing import Optional
from typing_extensions import Self

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
        with _WaitCounter("pytorch.filelock.enter").guard():
            result = super().__enter__()
        self.region_counter.__enter__()
        return result

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.region_counter.__exit__()
        with _WaitCounter("pytorch.filelock.exit").guard():
            # Returns nothing per
            # https://github.com/tox-dev/filelock/blob/57f488ff8fdc2193572efe102408fb63cfefe4e4/src/filelock/_api.py#L379
            super().__exit__(exc_type, exc_value, traceback)
        # Returns nothing per
        # https://github.com/pytorch/pytorch/blob/0f6bfc58a2cfb7a5c052bea618ab62becaf5c912/torch/csrc/monitor/python_init.cpp#L315
        return None
