from __future__ import annotations

from concurrent.futures import Future as _ThreadFuture
from datetime import timedelta
from typing import TYPE_CHECKING

from .._work import _FutureWork


if TYPE_CHECKING:
    from ._memory import MooncakeMemoryView, MooncakeRemoteBuffer
    from ._transport import MooncakeTransport


# Registered memory can still receive remote DMA after the last local transfer.
# Retain the owner until explicit unregister/close, not merely Work completion.
_live_transports: set[MooncakeTransport] = set()


class _MooncakeWork(_FutureWork):
    """Completion handle for a transfer settled by the transport's poller.

    A ``None`` future marks a zero-length no-op that is already complete.
    """

    def __init__(
        self,
        transport: MooncakeTransport,
        local: MooncakeMemoryView,
        remote: MooncakeRemoteBuffer,
        future: _ThreadFuture[None] | None,
        timeout: float | None,
    ) -> None:
        if future is None:
            future = _ThreadFuture()
            future.set_result(None)
        super().__init__(future, timeout)
        self._transport = transport
        self._buffers = (local, remote)

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        try:
            return super().wait(timeout)
        finally:
            # Settled transfers (success or failure) leave _pending here so a
            # later close() does not revisit and re-raise an observed failure.
            # A still-running transfer after a timeout stays pending.
            if self._future.done():
                self._transport._prune(id(self))
