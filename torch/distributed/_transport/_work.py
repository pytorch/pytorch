from __future__ import annotations

import asyncio
import math
from concurrent.futures import Future as _ThreadFuture, TimeoutError as _ThreadTimeout
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work


if TYPE_CHECKING:
    from collections.abc import Iterable


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and nonnegative")


class _FutureWork(Work):
    """Work settled by a ``concurrent.futures.Future``, possibly on another thread."""

    def __init__(self, future: _ThreadFuture[None], timeout: float | None) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._future = future
        self._timeout = timeout
        self._result: torch.futures.Future[Any] = torch.futures.Future()
        future.add_done_callback(self._settle)

    def _settle(self, future: _ThreadFuture[None]) -> None:
        error = future.exception()
        if error is None:
            self._result.set_result([])
        else:
            if not isinstance(error, Exception):
                error = RuntimeError(str(error))
            self._result.set_exception(error)

    def is_completed(self) -> bool:
        return self._future.done()

    def is_success(self) -> bool:
        return self.is_completed() and self.exception() is None

    def exception(self) -> BaseException | None:
        return self._future.exception() if self._future.done() else None

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        try:
            # Raises the operation's exception on failure; a timed-out operation
            # keeps running.
            self._future.result(timeout=seconds or self._timeout)
        except _ThreadTimeout:
            raise TimeoutError(
                "transport wait timed out; operation remains pending"
            ) from None
        return True

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        return self._result

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()


def _asyncio_future(work: Work) -> asyncio.Future[None]:
    loop = asyncio.get_running_loop()
    completion: asyncio.Future[None] = loop.create_future()

    def settle(error: BaseException | None) -> None:
        if completion.done():
            return
        if error is None:
            completion.set_result(None)
        else:
            completion.set_exception(error)

    def complete(future: torch.futures.Future[Any]) -> None:
        error: BaseException | None = None
        try:
            # Completion callbacks run only after the future is done; this
            # reads its result/error without waiting for unfinished host work.
            future.wait()
        except BaseException as failure:
            error = failure
        # Backend completion callbacks may run on a different thread. The
        # caller may also have closed its loop after cancellation or timeout.
        try:
            loop.call_soon_threadsafe(settle, error)
        except RuntimeError:
            # A cancelled waiter may close its loop before this callback runs.
            # Only suppress that race, not other event-loop failures.
            if not loop.is_closed():
                raise

    future = work.get_future()
    if future.done():
        # As in the completion callback, wait() only retrieves a finished result.
        try:
            future.wait()
        except BaseException as error:
            settle(error)
        else:
            settle(None)
    else:
        future.add_done_callback(complete)
    return completion


async def wait_all(works: Iterable[Work], *, timeout: float | None = None) -> None:
    """Await Work futures without blocking the asyncio loop.

    Timeout and cancellation stop waiting, not transfers. Retain buffers and
    wait again (or close the transport) before reusing them. Transfer errors and
    iterable errors are reported after draining submitted work, unless this wait
    is timed out or cancelled first. Each Work must support ``get_future``.
    This adapter does not select CUDA streams or establish consumer-stream
    ordering; callers must follow their backend's CUDA synchronization contract.
    """
    _validate_timeout(timeout)
    pending = []
    retained = []  # Keep each Work alive until its future has been awaited.
    error: BaseException | None = None
    try:
        for work in works:
            retained.append(work)
            pending.append(_asyncio_future(work))
    except BaseException as dispatch_error:
        error = dispatch_error
    try:
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=timeout)
            if unfinished:
                raise TimeoutError(
                    "transport wait timed out; operations remain pending"
                )
        errors = [future.exception() for future in pending]
        if error is not None:
            raise error
        for failure in errors:
            if failure is not None:
                raise failure
    finally:
        # These are local waiters, not backend futures. Retrieve any exceptions
        # already delivered, including when the caller stopped waiting early.
        for future in pending:
            if future.done():
                future.exception()
            else:
                future.cancel()
