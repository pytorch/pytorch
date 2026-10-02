from __future__ import annotations

import asyncio
import math
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch
    from torch.distributed import Work


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and nonnegative")


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
