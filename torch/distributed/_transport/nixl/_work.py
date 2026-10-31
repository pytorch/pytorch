from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work

from .._work import _validate_timeout


if TYPE_CHECKING:
    from ._memory import NIXLMemoryView, NIXLRemoteBuffer
    from ._transport import NIXLTransport


# Registered memory can still receive remote DMA after the last local transfer.
# Retain its owner until explicit unregister/close, not merely Work completion.
_live_transports: set[NIXLTransport] = set()


class _PollingWork(Work):
    """Work that resolves a future from backend status checks.

    Subclasses implement a nonblocking, thread-safe ``_poll`` and record a
    terminal error in ``_error``. A status query error the backend raises without
    finishing the transfer must leave the Work pending.
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
        # Future callbacks may synchronously call is_completed()/get_future() again.
        # Notify outside _future_lock to avoid deadlocking on that reentrancy.
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
            # Yielding alone would immediately poll again and consume a CPU core.
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


class _NIXLWork(_PollingWork):
    def __init__(
        self,
        transport: NIXLTransport,
        local: NIXLMemoryView,
        remote: NIXLRemoteBuffer,
        timeout: float,
    ) -> None:
        super().__init__(timeout)
        self._transport = transport
        self._buffers = (local, remote)
        self._descriptors: Any = None
        self._handle: Any = None
        self._state = "PROC"
        self._done = False

    def _poll(self) -> bool:
        if self._done:
            return True
        transport = self._transport
        if not transport._operation_lock.acquire(blocking=False):
            return False
        try:
            if self._done:
                return True
            if self._state == "PROC":
                try:
                    self._state = transport._agent.check_xfer_state(self._handle)
                except Exception as error:
                    # NIXL raises for a failed transfer. It also raises NOT_FOUND,
                    # without finishing the transfer, once a failed load dropped the
                    # peer's descriptors; the DMA may still run, so the Work stays
                    # pending and close() cannot finish.
                    if isinstance(error, transport._backend.nixlNotFoundError):
                        return False
                    self._state = "ERR"
                    if self._error is None:
                        self._error = error
            if self._state == "PROC":
                return False
            if self._state != "DONE":
                # The peer's loaded descriptors may be stale; reload them.
                transport._resetting.add(self._buffers[1].agent_name)
                if self._error is None:
                    self._error = RuntimeError(f"NIXL transfer failed: {self._state}")
            try:
                transport._agent.release_xfer_handle(self._handle)
                transport._transfers.pop(id(self))
            except BaseException as error:
                # Keep unreleased handles for close; disallow new requests so
                # an old Work's identity cannot be reused as a new handle key.
                # This is a one-way shutdown latch: close may be retried, but
                # the transport must never accept new transfers after this error.
                transport._closing = True
                if self._error is None:
                    self._error = error
            transport._pending.pop(id(self))
            transport._pending_peers[self._buffers[1].agent_name] -= 1
            # Lock-free readers use the recorded error and finalized bookkeeping.
            # Publish completion last; completed Works are never reset.
            self._done = True
            # Completion ends this Work, not the lifetime of exposed registrations.
            # unregister_memory()/close() remove the transport's retention root.
            return True
        finally:
            transport._operation_lock.release()
