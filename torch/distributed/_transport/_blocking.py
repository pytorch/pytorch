from __future__ import annotations

import threading
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work

from ._api import Transport
from ._work import _validate_timeout


if TYPE_CHECKING:
    from collections.abc import Callable


class _FutureWork(Work):
    def __init__(
        self, future: Future[int], queue: _WorkQueue, timeout: float | None = None
    ) -> None:
        super().__init__()
        self._future = future
        self._queue = queue
        self._timeout = timeout
        self._torch_future: torch.futures.Future[Any] = torch.futures.Future()
        future.add_done_callback(self._finish)

    def _result(self, timeout: float | None = None) -> None:
        try:
            status = self._future.result(timeout)
        except FutureTimeoutError as error:
            raise TimeoutError(str(error) or "transport wait timed out") from error
        if status != 0:
            raise RuntimeError(f"transport operation failed with status {status}")

    def _finish(self, future: Future[int]) -> None:
        try:
            self._result()
        except BaseException as error:
            if not isinstance(error, Exception):
                error = RuntimeError(str(error))
            self._torch_future.set_exception(error)
        else:
            self._torch_future.set_result([])

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        if (
            not self._future.done()
            and threading.get_ident() == self._queue.worker_ident
        ):
            raise RuntimeError("cannot wait for pending work from a transport callback")
        seconds = timeout.total_seconds()
        if seconds < 0:
            raise ValueError("timeout must be nonnegative")
        self._result(seconds or self._timeout)
        return True

    def is_completed(self) -> bool:
        return self._future.done()

    def is_success(self) -> bool:
        if not self._future.done():
            return False
        return self.exception() is None

    def exception(self) -> BaseException | None:
        if not self._future.done():
            return None
        try:
            self._result()
        except BaseException as error:
            return error
        return None

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        return self._torch_future

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()


class _WorkQueue:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._executor: ThreadPoolExecutor | None = None
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._pending = 0
        self._closed = False
        self.worker_ident: int | None = None

    def run(
        self,
        operation: Callable[[], int],
        device: torch.device,
        *,
        async_op: bool,
        timeout: float | None = None,
    ) -> int | Work:
        _validate_timeout(timeout)
        capturing = False
        if device.type == "cuda":
            with torch.cuda.device(device):
                capturing = torch.cuda.is_current_stream_capturing()
        if capturing and async_op:
            raise RuntimeError("async transport operations cannot be captured")
        if not async_op and threading.get_ident() == self.worker_ident:
            raise RuntimeError(
                "cannot run a synchronous transfer from a transport callback"
            )
        with self._condition:
            if self._closed:
                raise RuntimeError("transport is closed")
            if capturing and self._pending:
                raise RuntimeError(
                    "wait for pending transport work before CUDA graph capture"
                )
            if capturing:
                self._pending += 1
                future = None
            else:
                ready = None
                if device.type == "cuda":
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(device))
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="transport"
                    )
                self._pending += 1
                try:
                    future = self._executor.submit(
                        self._execute, operation, device, ready
                    )
                except BaseException:
                    self._pending -= 1
                    self._condition.notify_all()
                    raise
        if future is not None:
            if async_op:
                return _FutureWork(future, self, timeout)
            try:
                return future.result(timeout)
            except FutureTimeoutError as error:
                if future.done():
                    return future.result()
                raise TimeoutError(
                    "transport wait timed out; operation remains pending"
                ) from error
        try:
            return operation()
        finally:
            self._finished()

    def _execute(
        self,
        operation: Callable[[], int],
        device: torch.device,
        ready: torch.cuda.Event | None,
    ) -> int:
        self.worker_ident = threading.get_ident()
        try:
            if ready is None:
                return operation()
            with torch.cuda.device(device):
                ready.synchronize()
                stream = self._streams.get(device)
                if stream is None:
                    stream = torch.cuda.Stream(device=device)
                    self._streams[device] = stream
                with torch.cuda.stream(stream):
                    try:
                        return operation()
                    finally:
                        stream.synchronize()
        finally:
            self._finished()

    def _finished(self) -> None:
        with self._condition:
            self._pending -= 1
            self._condition.notify_all()

    def close(self, timeout: float | None = None) -> None:
        _validate_timeout(timeout)
        if threading.get_ident() == self.worker_ident:
            raise RuntimeError("cannot close a transport from its worker")
        with self._condition:
            self._closed = True
            if not self._condition.wait_for(lambda: self._pending == 0, timeout):
                raise TimeoutError("transport close timed out; resources remain live")
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=timeout is None)
        self.worker_ident = None
        self._streams.clear()


class _BlockingTransport(Transport):
    """Executor adapter for prototype backends with blocking transfer APIs."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__(device)
        self._work_queue = _WorkQueue()

    def _run_transfer(
        self,
        operation: Callable[[], int],
        device: torch.device,
        *,
        async_op: bool,
        timeout: float | None = None,
    ) -> int | Work:
        return self._work_queue.run(
            operation, device, async_op=async_op, timeout=timeout
        )

    def _close_work(self, timeout: float | None = None) -> None:
        self._work_queue.close(timeout)
