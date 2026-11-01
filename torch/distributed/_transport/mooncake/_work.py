from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work
from torch.distributed._transport._work import _validate_timeout


if TYPE_CHECKING:
    from ._memory import MooncakeMemoryView, MooncakeRemoteBuffer
    from ._transport import MooncakeTransport


# Exposed memory must survive even when Python callers drop the transport/Work.
# Only explicit unregistration or successful close removes this retention root.
_live_transports: set[MooncakeTransport] = set()


class MooncakeWork(Work):
    def __init__(
        self,
        transport: MooncakeTransport,
        local: MooncakeMemoryView,
        remote: MooncakeRemoteBuffer,
        timeout: float,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._buffers = (local, remote)
        self._timeout = timeout
        self._operation: Any = None
        self._done = False
        self._error: Exception | None = None
        self._future: torch.futures.Future[Any] = torch.futures.Future()
        self._future_lock = threading.Lock()
        self._notified = False
        self._progress: asyncio.Task[None] | None = None

    def is_completed(self) -> bool:
        owner = self._transport
        if not owner._lock.acquire(blocking=False):
            return False
        try:
            if not self._done:
                try:
                    done, error = self._operation.poll()
                except Exception as error:
                    # A query failure does not establish physical completion.
                    self._error = error
                    return False
                if not done:
                    return False
                if error and self._error is None:
                    self._error = RuntimeError(error)
                self._operation = None
                self._done = True
                owner._pending.pop(id(self), None)
        finally:
            owner._lock.release()
        with self._future_lock:
            notify = not self._notified
            self._notified = True
        # Callbacks can re-enter Work methods; never notify under either lock.
        if notify:
            if self._error is None:
                self._future.set_result([])
            else:
                self._future.set_exception(self._error)
        return True

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        deadline = time.monotonic() + (seconds or self._timeout)
        while not self.is_completed():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "transport wait timed out; operation remains pending"
                )
            # The native handle offers polling only; bound CPU usage between queries.
            time.sleep(0.001)
        if self._error is not None:
            raise self._error
        return True

    async def _drive(self) -> None:
        while not self.is_completed():
            await asyncio.sleep(0.001)

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        if not self.is_completed():
            loop = asyncio.get_running_loop()
            if self._progress is None or self._progress.done():
                self._progress = loop.create_task(self._drive())
        return self._future

    def is_success(self) -> bool:
        return self.is_completed() and self._error is None

    def exception(self) -> Exception | None:
        return self._error if self.is_completed() else None

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()
