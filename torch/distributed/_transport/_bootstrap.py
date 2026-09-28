from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TYPE_CHECKING

import torch.distributed as dist

from ._registry import new_transport
from ._work import _validate_timeout


if TYPE_CHECKING:
    import torch

    from ._api import Transport


@dataclass
class _RankBootstrap:
    store: dist.Store
    rank: int
    peer_rank: int
    timeout: float

    @classmethod
    def create(
        cls,
        backend: str,
        peer_rank: int,
        store: dist.Store,
        rank: int | None,
        tag: str | None,
        timeout: float,
    ) -> _RankBootstrap:
        _validate_timeout(timeout)
        if timeout is None:
            raise ValueError("bootstrap_timeout must be finite")
        if not isinstance(tag, str) or not tag:
            raise ValueError("bootstrap_tag must be a nonempty, unique string")
        rank = dist.get_rank() if rank is None else rank
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if type(peer_rank) is not int or peer_rank < 0:
            raise ValueError("peer_rank must be a nonnegative integer")
        if peer_rank == rank:
            raise ValueError("peer_rank must differ from the local rank")
        prefix = json.dumps(
            ["transport", backend, tag, min(rank, peer_rank), max(rank, peer_rank)]
        )
        store = dist.PrefixStore(prefix, store)
        # Never consume stale endpoint data when a tag is accidentally reused.
        if store.add(f"claim/{rank}", 1) != 1:
            raise ValueError("bootstrap_tag was already used for this rank pair")
        return cls(store, rank, peer_rank, timeout)

    def connect(self, transport: Transport) -> None:
        deadline = time.monotonic() + self.timeout

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("transport bootstrap timed out")
            return seconds

        try:
            endpoint = transport.bind(timeout=remaining())
            nonce = uuid.uuid4().hex
            self.store.set(
                f"endpoint/{nonce}", base64.b64encode(endpoint).decode("ascii")
            )
            self.store.set(str(self.rank), nonce)
            self.store.wait([str(self.peer_rank)], timedelta(seconds=remaining()))
            peer_nonce = self.store.get(str(self.peer_rank)).decode("ascii")
            endpoint = base64.b64decode(
                self.store.get(f"endpoint/{peer_nonce}"), validate=True
            )
            # Confirm the peer observed this attempt, not a stale publication.
            self.store.set(f"ack/{nonce}/{peer_nonce}", "ready")
            self.store.wait(
                [f"ack/{peer_nonce}/{nonce}"], timedelta(seconds=remaining())
            )
            transport.connect(endpoint, timeout=remaining())
        except BaseException as error:
            try:
                transport.close(timeout=max(0.0, deadline - time.monotonic()))
            except Exception as cleanup_error:
                error.add_note(f"transport bootstrap cleanup failed: {cleanup_error}")
            raise


def new_transport_rank(
    backend: str,
    device: torch.device | str | None = None,
    *,
    store: dist.Store,
    peer_rank: int,
    rank: int | None = None,
    bootstrap_tag: str,
    bootstrap_timeout: float = 30.0,
    **kwargs: Any,
) -> Transport:
    """Construct, bind, and connect a transport through a supplied Store.

    Args:
        backend: Registered transport backend name.
        device: Optional tensor device restriction, as in ``new_transport``.
        store: Shared rendezvous Store, such as ``torch.distributed.TCPStore``.
            The caller owns its lifetime and supplies it to both peers. No process
            group Store is used and no collectives are performed.
        peer_rank: Nonnegative integer identifying the peer in this Store namespace.
        rank: Local identity; defaults to ``torch.distributed.get_rank()``. An
            explicit override requires no initialized process group. Identities
            need not be contiguous, but must be distinct for the two peers.
        bootstrap_tag: Nonempty unique name per rank pair for the Store lifetime,
            including failed attempts. Both peers must use the same tag and call
            concurrently with reciprocal rank/peer_rank values.
        bootstrap_timeout: Bind, endpoint exchange, and connect wait budget in
            seconds. Native calls and Store network operations may exceed this
            budget. Does not change the supplied Store's timeout.
        **kwargs: Forwarded to ``new_transport`` and the backend constructor.

    Example::

        from torch.distributed import TCPStore
        from torch.distributed._transport import new_transport_rank

        # Run on two processes, assigning rank=0 and rank=1 respectively.
        # The Store server must remain alive while both peers bootstrap.
        store = TCPStore("server-host", 29501, world_size=2, is_master=(rank == 0))
        transport = new_transport_rank(
            "nixl",
            "cpu",
            store=store,
            rank=rank,
            peer_rank=1 - rank,
            bootstrap_tag="checkpoint-channel-0",
        )
        # Exchange registered-memory descriptors separately. Coordinate with
        # peers before unregistering memory or closing the transport.

    Duplicate local claims fail rather than reuse an endpoint. A nonce handshake
    prevents accepting stale peer publications. Failed attempts consume their tag;
    retry with a new tag. Construction/connection failures do not release Store
    keys. Bootstrap failures attempt to close the partially created transport.
    """
    bootstrap = _RankBootstrap.create(
        backend.lower(), peer_rank, store, rank, bootstrap_tag, bootstrap_timeout
    )
    transport = new_transport(backend, device, **kwargs)
    bootstrap.connect(transport)
    return transport
