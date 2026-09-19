from __future__ import annotations

import asyncio
import math
import threading
import time
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work


if TYPE_CHECKING:
    from collections.abc import Iterable


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and nonnegative")


async def wait_all(works: Iterable[Work], *, timeout: float | None = None) -> None:
    """Poll Work completion without an executor or blocking the asyncio loop.

    Timeout and cancellation stop waiting, not transfers. Retain buffers and
    wait again (or close the transport) before reusing them. Transfer errors and
    iterable errors are reported after draining submitted work, unless this wait
    is timed out or cancelled first. Polling checks must be nonblocking.
    """
    _validate_timeout(timeout)
    deadline = None if timeout is None else time.monotonic() + timeout
    pending = []
    error: BaseException | None = None
    try:
        pending.extend(works)
    except BaseException as dispatch_error:
        error = dispatch_error
    while pending:
        remaining = []
        for work in pending:
            if not work.is_completed():
                remaining.append(work)
                continue
            try:
                work.wait()
            except BaseException as work_error:
                if error is None:
                    error = work_error
        pending = remaining
        if not pending:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("transport wait timed out; operations remain pending")
        await asyncio.sleep(0.001)
    if error is not None:
        raise error


class _PollingWork(Work):
    """Work driven by native status checks, never a Python worker thread.

    Subclasses implement a nonblocking, thread-safe ``_poll`` and record a
    terminal error in ``_error``. A failed status query must not report completion
    unless it establishes that the backend has stopped accessing memory.
    """

    def __init__(self, timeout: float | None = None) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._timeout = timeout
        self._error: BaseException | None = None
        self._future: torch.futures.Future[Any] = torch.futures.Future()
        self._future_lock = threading.Lock()
        self._future_completed = False
        self._progress_task: asyncio.Task[None] | None = None

    def _poll(self) -> bool:
        raise NotImplementedError

    def is_completed(self) -> bool:
        if not self._poll():
            return False
        with self._future_lock:
            notify = not self._future_completed
            self._future_completed = True
        # Callbacks may reenter the transport: never invoke them under its lock.
        if notify:
            if self._error is None:
                self._future.set_result([])
            else:
                error = self._error
                if not isinstance(error, Exception):
                    error = RuntimeError(str(error))
                self._future.set_exception(error)
        return True

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        seconds = seconds or self._timeout
        deadline = None if seconds is None else time.monotonic() + seconds
        while not self.is_completed():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "transport wait timed out; operation remains pending"
                )
            time.sleep(0.001)
        if self._error is not None:
            raise self._error
        return True

    def is_success(self) -> bool:
        return self.is_completed() and self._error is None

    def exception(self) -> BaseException | None:
        return self._error if self.is_completed() else None

    async def _drive_future(self) -> None:
        while not self.is_completed():
            await asyncio.sleep(0.001)

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        if self.is_completed():
            return self._future
        loop = asyncio.get_running_loop()
        if self._progress_task is None or self._progress_task.done():
            self._progress_task = loop.create_task(self._drive_future())
        return self._future

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()
