from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch.distributed as dist

from ._registry import new_transport
from ._work import _validate_timeout


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch

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
    peer connects to the other's Nth call, including failed calls. One Store can
    therefore connect a rank to every other rank. Bootstrap failures attempt to
    close the partially created transport.
    """
    if not isinstance(bootstrap_timeout, timedelta):
        bootstrap_timeout = timedelta(seconds=bootstrap_timeout)
    timeout = bootstrap_timeout.total_seconds()
    _validate_timeout(timeout)
    rank = dist.get_rank() if rank is None else rank
    store = dist.distributed_c10d._get_default_store() if store is None else store
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if type(peer_rank) is not int or peer_rank < 0:
        raise ValueError("peer_rank must be a nonnegative integer")
    if peer_rank == rank:
        raise ValueError("peer_rank must differ from the local rank")
    pair = f"transport/{backend.lower()}/{min(rank, peer_rank)}/{max(rank, peer_rank)}"
    attempt = store.add(f"{pair}/attempts/{rank}", 1)
    store = dist.PrefixStore(f"{pair}/{attempt}", store)
    transport = new_transport(backend, device, **kwargs)
    try:
        # Store.set accepts bytes; the stub only declares str.
        # pyrefly: ignore [bad-argument-type]
        store.set(str(rank), transport.bind(timeout=timeout))
        store.wait([str(peer_rank)], bootstrap_timeout)
        transport.connect(store.get(str(peer_rank)), timeout=timeout)
    except BaseException:
        try:
            transport.close(timeout=timeout)
        except Exception as cleanup_error:
            # Exception notes require Python 3.11.
            logger.warning("transport bootstrap cleanup failed", exc_info=cleanup_error)
        raise

    return transport
