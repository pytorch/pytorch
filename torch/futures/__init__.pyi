# Stub file for torch.futures
from collections.abc import Callable
from typing import Generic, TypeVar

import torch
from torch._C import Future as _CFuture

__all__ = ["Future", "collect_all", "wait_all"]

T = TypeVar("T")
S = TypeVar("S")


class Future(_CFuture[T], Generic[T]):
    """
    Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
    execution of a callable.
    """

    def __init__(
        self, *, devices: list[int | str | torch.device] | None = None
    ) -> None: ...
    def done(self) -> bool: ...
    def wait(self) -> T: ...
    def value(self) -> T: ...
    def then(self, callback: Callable[[Future[T]], S]) -> Future[S]: ...
    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None: ...
    def set_result(self, result: T) -> None: ...
    def set_exception(self, result: BaseException) -> None: ...


def collect_all(futures: list[Future[T]]) -> Future[list[Future[T]]]: ...
def wait_all(futures: list[Future[T]]) -> list[T]: ...

