from __future__ import annotations

from concurrent.futures import Future as _ThreadFuture, TimeoutError as _ThreadTimeout
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import Work

from .._work import _validate_timeout


if TYPE_CHECKING:
    from ._memory import MooncakeMemoryView, MooncakeRemoteBuffer
    from ._transport import MooncakeTransport


# Registered memory can still receive remote DMA after the last local transfer.
# Retain the owner until explicit unregister/close, not merely Work completion.
_live_transports: set[MooncakeTransport] = set()


class _MooncakeWork(Work):
    """Completion handle for a transfer run on the transport's executor.

    Mooncake's Python transfer APIs block until the transfer finishes, so there
    is no nonblocking status to poll. Each async transfer runs its blocking call
    on a worker thread; this Work tracks that thread's future. A ``None`` future
    marks a zero-length no-op that is already complete.
    """

    def __init__(
        self,
        transport: MooncakeTransport,
        local: MooncakeMemoryView,
        remote: MooncakeRemoteBuffer,
        future: _ThreadFuture[None] | None,
        timeout: float | None,
    ) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._transport = transport
        self._buffers = (local, remote)
        self._future = future
        self._timeout = timeout
        self._result: torch.futures.Future[Any] = torch.futures.Future()
        if future is None:
            self._result.set_result([])
        else:
            future.add_done_callback(self._settle)

    def _settle(self, future: _ThreadFuture[None]) -> None:
        # Runs on the worker thread once the blocking transfer returns.
        error = future.exception()
        if error is None:
            self._result.set_result([])
        else:
            if not isinstance(error, Exception):
                error = RuntimeError(str(error))
            self._result.set_exception(error)

    def is_completed(self) -> bool:
        return self._future is None or self._future.done()

    def is_success(self) -> bool:
        return self.is_completed() and self.exception() is None

    def exception(self) -> BaseException | None:
        if self._future is None or not self._future.done():
            return None
        return self._future.exception()

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        if self._future is None:
            return True
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        seconds = seconds or self._timeout
        try:
            # Raises the transfer's exception on failure; the transfer keeps
            # running on timeout, so buffers must stay retained until it settles.
            self._future.result(timeout=seconds)
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
