from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, TYPE_CHECKING

from ._api import _exchange, _pair_store
from ._registry import new_transport
from ._work import _validate_timeout


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from ._api import Transport


def new_transport_rank(
    backend: str,
    device: torch.device | str | None = None,
    *,
    store: dist.Store | None = None,
    peer_rank: int,
    rank: int | None = None,
    bootstrap_timeout: float | timedelta = 30.0,
    **kwargs: Any,
) -> Transport:
    """Construct, bind, and connect a transport through a Store.

    Args:
        backend: Registered transport backend name.
        device: Optional tensor device restriction, as in ``new_transport``.
        store: Shared rendezvous Store, such as ``torch.distributed.TCPStore``;
            defaults to the default process group's Store. No collectives are
            performed.
        peer_rank: Nonnegative integer identifying the peer in this Store namespace.
        rank: Local identity; defaults to ``torch.distributed.get_rank()``. An
            explicit override requires no initialized process group. Identities
            need not be contiguous, but must be distinct for the two peers.
        bootstrap_timeout: Timeout for each of bind, the Store waits, connect,
            and cleanup, in seconds or as a ``timedelta``. Does not change the
            Store's timeout.
        **kwargs: Forwarded to ``new_transport`` and the backend constructor.

    Example::

        import torch.distributed as dist
        from torch.distributed._transport import new_transport_rank

        dist.init_process_group("gloo")
        # Pair ranks 0 and 1; other ranks need not participate.
        transport = new_transport_rank(
            "nixl",
            "cpu",
            peer_rank=1 - dist.get_rank(),
        )
        # Exchange registered-memory descriptors separately. Coordinate with
        # peers before unregistering memory or closing the transport.

    Calls are matched in order per rank pair and backend: the Nth call on each
    peer connects to the other's Nth call, including calls that failed after
    validating ranks, such as backend construction errors. One Store can
    therefore connect a rank to every other rank. To connect one transport to several ranks, use
    ``Transport.connect_rank``. Bootstrap failures attempt to close the
    partially created transport.
    """
    if not isinstance(bootstrap_timeout, timedelta):
        bootstrap_timeout = timedelta(seconds=bootstrap_timeout)
    timeout = bootstrap_timeout.total_seconds()
    _validate_timeout(timeout)
    store, rank = _pair_store(f"transport/{backend.lower()}", store, rank, peer_rank)
    transport = new_transport(backend, device, **kwargs)
    try:
        _exchange(transport, store, rank, peer_rank, bootstrap_timeout)
    except BaseException:
        try:
            transport.close(timeout=timeout)
        except Exception as cleanup_error:
            # Exception notes require Python 3.11.
            logger.warning("transport bootstrap cleanup failed", exc_info=cleanup_error)
        raise

    return transport
