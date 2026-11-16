from __future__ import annotations

import asyncio
import math
import threading
import time
import weakref
from concurrent.futures import Future as _ThreadFuture, TimeoutError as _ThreadTimeout
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and nonnegative")


# Listeners wait_all registers on a _FutureWork's thread future. Unlike torch
# future callbacks they can be removed, so a timed-out wait leaves nothing on a
# future that may stay pending indefinitely, such as the one shared by
# notification waits.
_listeners: weakref.WeakKeyDictionary[
    _ThreadFuture[None], dict[object, Callable[[_ThreadFuture[None]], None]]
] = weakref.WeakKeyDictionary()
_listeners_lock = threading.Lock()


def _exception(future: _ThreadFuture[None]) -> Exception | None:
    error = future.exception()
    if error is None or isinstance(error, Exception):
        return error
    return RuntimeError(str(error))


def _notify_listeners(future: _ThreadFuture[None]) -> None:
    with _listeners_lock:
        listeners = list(_listeners.pop(future, {}).values())
    for listener in listeners:
        listener(future)


def _future_error(future: torch.futures.Future[Any]) -> Exception | None:
    try:
        future.wait()
    except Exception as error:
        return error
    return None


def _complete(result: torch.futures.Future[Any], error: BaseException | None) -> None:
    if error is None:
        result.set_result([])
    else:
        result.set_exception(error)


class _FutureWork(Work):
    """Work settled by a ``concurrent.futures.Future``, possibly on another thread."""

    def __init__(self, future: _ThreadFuture[None], timeout: float | None) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._future = future
        self._timeout = timeout
        # Created on first get_future, so Works that are only waited on, such as
        # timed-out notification waits sharing one future, add no callbacks to it.
        self._result: torch.futures.Future[Any] | None = None
        self._result_lock = threading.Lock()

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
        with self._result_lock:
            result = self._result
            if result is None:
                result = self._result = torch.futures.Future()
                self._future.add_done_callback(
                    lambda future: _complete(result, _exception(future))
                )
        return result

    def _add_listener(
        self, listener: Callable[[_ThreadFuture[None]], None]
    ) -> Callable[[], None]:
        """Call ``listener(future)`` once settled; return a function removing it."""
        key = object()
        with _listeners_lock:
            listeners = _listeners.get(self._future)
            first = listeners is None
            if listeners is None:
                listeners = _listeners[self._future] = {}
            listeners[key] = listener
        if first:
            # Outside the lock: runs inline if the future is already done.
            self._future.add_done_callback(_notify_listeners)

        def remove() -> None:
            with _listeners_lock:
                _listeners.get(self._future, {}).pop(key, None)

        return remove

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()


class _BatchWork(Work):
    """Work for a batch, terminal once every child Work is terminal."""

    def __init__(
        self,
        works: list[Work],
        dispatch_error: Exception | None,
        timeout: float | None,
    ) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._works = works
        self._dispatch_error = dispatch_error
        self._timeout = timeout
        self._result: torch.futures.Future[Any] | None = None
        self._result_lock = threading.Lock()

    def is_completed(self) -> bool:
        return all(work.is_completed() for work in self._works)

    def is_success(self) -> bool:
        return self.is_completed() and self.exception() is None

    def exception(self) -> BaseException | None:
        if not self.is_completed():
            return None
        if self._dispatch_error is not None:
            return self._dispatch_error
        return next(
            (error for work in self._works if (error := work.exception()) is not None),
            None,
        )

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        seconds = seconds or self._timeout
        deadline = None if seconds is None else time.monotonic() + seconds
        error = None
        for work in self._works:
            # A zero timedelta selects the child's own default, so the time
            # left is never passed down as zero.
            remaining = timedelta(0)
            if deadline is not None:
                remaining = timedelta(seconds=max(deadline - time.monotonic(), 1e-6))
            try:
                if not work.wait(remaining):
                    return False
            except Exception:
                # A failed child is terminal; keep waiting for the others.
                if not work.is_completed():
                    raise
                if error is None:
                    error = work.exception()
        if self._dispatch_error is not None:
            raise self._dispatch_error
        if error is not None:
            raise error
        return True

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        # Asked on every call: a child may drive its future from the calling
        # event loop, as NIXL's does, and that driver dies with the loop.
        futures = [work.get_future() for work in self._works]
        with self._result_lock:
            if self._result is not None:
                return self._result
            result = self._result = torch.futures.Future()
        # torch.futures.collect_all completes as soon as a child future holds a
        # C++ error, such as one raised in a then() callback, while other
        # children may still be running. Count completions instead.
        remaining = len(futures)
        lock = threading.Lock()

        def settle(_: torch.futures.Future[Any]) -> None:
            nonlocal remaining
            with lock:
                remaining -= 1
                if remaining > 0:
                    return
            error = self._dispatch_error
            if error is None:
                error = next(
                    (e for f in futures if (e := _future_error(f)) is not None), None
                )
            _complete(result, error)

        if not futures:
            result.set_result([])
        for future in futures:
            future.add_done_callback(settle)
        return result

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()


def _asyncio_future(
    work: Work, cleanups: list[Callable[[], None]]
) -> asyncio.Future[None]:
    loop = asyncio.get_running_loop()
    completion: asyncio.Future[None] = loop.create_future()

    def settle(error: BaseException | None) -> None:
        if completion.done():
            return
        if error is None:
            completion.set_result(None)
        else:
            completion.set_exception(error)

    def deliver(error: BaseException | None) -> None:
        # Backend completion callbacks may run on a different thread. The
        # caller may also have closed its loop after cancellation or timeout.
        try:
            loop.call_soon_threadsafe(settle, error)
        except RuntimeError:
            # A cancelled waiter may close its loop before this callback runs.
            # Only suppress that race, not other event-loop failures.
            if not loop.is_closed():
                raise

    if isinstance(work, _FutureWork):
        # wait_all removes this listener when it stops waiting; see _listeners.
        cleanups.append(work._add_listener(lambda f: deliver(_exception(f))))
        return completion

    def complete(future: torch.futures.Future[Any]) -> None:
        error: BaseException | None = None
        try:
            # Completion callbacks run only after the future is done; this
            # reads its result/error without waiting for unfinished host work.
            future.wait()
        except BaseException as failure:
            error = failure
        deliver(error)

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
    cleanups: list[Callable[[], None]] = []
    error: BaseException | None = None
    try:
        for work in works:
            retained.append(work)
            pending.append(_asyncio_future(work, cleanups))
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
        for cleanup in cleanups:
            cleanup()
        # These are local waiters, not backend futures. Retrieve any exceptions
        # already delivered, including when the caller stopped waiting early.
        for future in pending:
            if future.done():
                future.exception()
            else:
                future.cancel()
