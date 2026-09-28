from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _get_process_group_store,
)

from ._work import _validate_timeout


if TYPE_CHECKING:
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
        group: dist.ProcessGroup | None,
        tag: str | None,
        timeout: float,
    ) -> _RankBootstrap:
        _validate_timeout(timeout)
        if timeout is None:
            raise ValueError("bootstrap_timeout must be finite")
        if not isinstance(tag, str) or not tag:
            raise ValueError("bootstrap_tag must be a nonempty, unique string")
        if not dist.is_initialized():
            raise RuntimeError(
                "rank bootstrapping requires an initialized process group"
            )
        group = _get_default_group() if group is None else group
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        if rank < 0:
            raise ValueError("caller is not a member of the bootstrap group")
        if type(peer_rank) is not int or not 0 <= peer_rank < size:
            raise ValueError("peer_rank must be a rank within the bootstrap group")
        if peer_rank == rank:
            raise ValueError("peer_rank must differ from the local rank")
        prefix = json.dumps(
            ["transport", backend, tag, min(rank, peer_rank), max(rank, peer_rank)]
        )
        store = dist.PrefixStore(prefix, _get_process_group_store(group))
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
