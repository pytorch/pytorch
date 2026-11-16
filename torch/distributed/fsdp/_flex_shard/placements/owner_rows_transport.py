# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Owner rows through equal chunks: symmetric memory inside a host, NCCL's
native all-gather and reduce-scatter across the mesh.

A bucket whose groups each live whole on one owner gathers its owner rows back
to back (see ``mixed_bucket._owner_rows_plan``). The rows have their owners'
lengths, so they're uneven, and the grouped broadcasts and reduces that take
them as they are run well below NCCL's native collectives across hosts. Here
the rows are cut into equal chunks instead, one per rank. Before the
all-gather, each owner writes the slices of its row that lie in other ranks'
chunks straight into those ranks' buffers, which are in symmetric memory and
mapped by every GPU of the host. Bytes for another host go through relays,
split evenly over the ranks of both hosts, and NCCL carries them between
hosts. The reduce runs the native reduce-scatter, then the same moves in
reverse.
"""

from __future__ import annotations

import functools
import itertools
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..utils import mesh_rank_hosts
from .utils import foreach_copy_, reduce_scatter_grads


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.distributed.device_mesh import DeviceMesh


# Chunk sizes, rail splits and relayed slices round up to this many elements,
# 256 bytes in BF16.
_ALIGN = 128


def _align_up(numel: int) -> int:
    return -(-numel // _ALIGN) * _ALIGN


class OwnerRowsMove(NamedTuple):
    """``numel`` elements starting at ``start`` in the gathered buffer, from
    mesh rank ``src`` to mesh rank ``dst`` (the other way for the reduce)."""

    src: int
    dst: int
    start: int
    numel: int


@dataclass(frozen=True)
class OwnerRowsTransportPlan:
    """How one bucket's owner rows reach equal chunks, in mesh ranks.

    The gathered buffer holds the rows back to back, padded to
    ``gathered_numel``; rank r holds chunk ``[r * chunk_numel, (r + 1) *
    chunk_numel)``. Each owner keeps the part of its row inside its own chunk.
    For the gather, the rest moves:
    - ``owner_moves``, inside the owner's host: to the rank holding that
      chunk, or to a relay when that rank is on another host;
    - ``host_moves``, between hosts: rank j of one host to rank j of the
      other, each host pair's bytes split evenly over their ranks;
    - ``relay_moves``, inside the holder's host: relay to holder.
    The reduce makes the same moves in reverse. Each rank's buffer holds its
    chunk, then the slices it relays (see ``buffer_offset``)."""

    chunk_numel: int
    gathered_numel: int
    owner_moves: tuple[OwnerRowsMove, ...]
    host_moves: tuple[OwnerRowsMove, ...]
    relay_moves: tuple[OwnerRowsMove, ...]
    relay_offsets: dict[tuple[int, int], int]
    buffer_numel: int

    def buffer_offset(self, rank: int, start: int) -> int:
        """Where the slice starting at ``start`` sits in ``rank``'s buffer:
        at its place inside the rank's chunk, else after the chunk."""
        if rank * self.chunk_numel <= start < (rank + 1) * self.chunk_numel:
            return start - rank * self.chunk_numel
        return self.relay_offsets[(rank, start)]


@functools.cache
def plan_owner_rows_transport(
    row_numels: tuple[int, ...], hosts: tuple[int, ...]
) -> OwnerRowsTransportPlan:
    """The moves for owner rows of ``row_numels`` on mesh ranks on ``hosts``
    (see ``mesh_rank_hosts``). Every rank computes the same plan."""
    world_size = len(row_numels)
    if len(hosts) != world_size:
        raise ValueError(f"hosts must give each of {world_size} ranks a host.")
    chunk = _align_up(-(-sum(row_numels) // world_size))
    members: dict[int, list[int]] = defaultdict(list)
    for rank, host in enumerate(hosts):
        members[host].append(rank)

    owner_moves: list[OwnerRowsMove] = []
    host_moves: list[OwnerRowsMove] = []
    relay_moves: list[OwnerRowsMove] = []
    leaving: dict[tuple[int, int], list[OwnerRowsMove]] = defaultdict(list)
    for owner, (row_start, row_numel) in enumerate(
        zip(itertools.accumulate(row_numels, initial=0), row_numels)
    ):
        position, end = row_start, row_start + row_numel
        while position < end:
            holder = position // chunk
            stop = min(end, (holder + 1) * chunk)
            if holder != owner:
                move = OwnerRowsMove(owner, holder, position, stop - position)
                if hosts[owner] == hosts[holder]:
                    owner_moves.append(move)
                else:
                    leaving[(hosts[owner], hosts[holder])].append(move)
            position = stop
    for (src_host, dst_host), moves in leaving.items():
        # Rank j of each host relays the j-th part, one network card each.
        rails = min(len(members[src_host]), len(members[dst_host]))
        part = _align_up(-(-sum(move.numel for move in moves) // rails))
        sent = 0
        for owner, holder, start, numel in moves:
            while numel:
                rail = sent // part
                take = min(numel, (rail + 1) * part - sent)
                relay_out = members[src_host][rail]
                relay_in = members[dst_host][rail]
                if relay_out != owner:
                    owner_moves.append(OwnerRowsMove(owner, relay_out, start, take))
                host_moves.append(OwnerRowsMove(relay_out, relay_in, start, take))
                if relay_in != holder:
                    relay_moves.append(OwnerRowsMove(relay_in, holder, start, take))
                start, numel, sent = start + take, numel - take, sent + take

    relay_offsets: dict[tuple[int, int], int] = {}
    relay_numels: dict[int, int] = defaultdict(int)

    def hold(rank: int, start: int, numel: int) -> None:
        if rank * chunk <= start < (rank + 1) * chunk or (rank, start) in relay_offsets:
            return
        relay_offsets[(rank, start)] = chunk + relay_numels[rank]
        relay_numels[rank] += _align_up(numel)

    for move in owner_moves:
        hold(move.dst, move.start, move.numel)
    for move in host_moves:
        hold(move.src, move.start, move.numel)
        hold(move.dst, move.start, move.numel)
    for move in relay_moves:
        hold(move.src, move.start, move.numel)
    return OwnerRowsTransportPlan(
        chunk_numel=chunk,
        gathered_numel=world_size * chunk,
        owner_moves=tuple(owner_moves),
        host_moves=tuple(host_moves),
        relay_moves=tuple(relay_moves),
        relay_offsets=relay_offsets,
        buffer_numel=chunk + max(relay_numels.values(), default=0),
    )


@dataclass
class SymmetricBuffer:
    """A buffer in symmetric memory and every host rank's buffer mapped here,
    by mesh rank. ``readers`` are asynchronous collectives that may still read
    it; ``views`` caches each bucket layout's views into it (see
    ``gather_owner_rows``), dropped with the buffer when it grows."""

    tensor: torch.Tensor
    handle: Any
    peers: dict[int, torch.Tensor]
    readers: list[dist.Work] = field(default_factory=list)
    views: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def barrier(self) -> None:
        """Wait on this stream until every rank of the host reaches its
        barrier: its earlier writes into others' buffers are then visible."""
        self.handle.barrier(channel=0)

    def wait_readers(self) -> None:
        """Make this stream wait for the collectives still reading it."""
        for work in self.readers:
            work.wait()
        self.readers.clear()


class OwnerRowsTransportBuffers:
    """A mesh's symmetric buffers on this rank, by use and dtype, and the gloo
    group of this rank's host that their rendezvous runs over.

    A gloo group needs no device memory; an eagerly built NCCL group would take
    about 230 MiB. Each buffer grows to the largest request. A rendezvous is
    collective over the host, so every rank of the host must request a buffer
    at the same point, as SPMD code does for each bucket."""

    def __init__(self, mesh: DeviceMesh, hosts: Sequence[int]) -> None:
        if mesh.ndim != 1:
            raise ValueError(f"Owner rows need a 1-D mesh, got {mesh.ndim} dims.")
        if len(hosts) != mesh.size():
            raise ValueError(f"hosts must give each of {mesh.size()} ranks a host.")
        self._global_ranks = mesh.mesh.flatten().tolist()
        rank = mesh.get_local_rank()
        self.host_ranks = tuple(
            other for other, host in enumerate(hosts) if host == hosts[rank]
        )
        self.group = dist.new_group(
            [self._global_ranks[other] for other in self.host_ranks],
            backend="gloo",
            use_local_synchronization=True,
        )
        self._buffers: dict[tuple[str, torch.dtype], SymmetricBuffer] = {}

    def get(
        self, use: str, dtype: torch.dtype, numel: int, device: torch.device
    ) -> SymmetricBuffer:
        """This rank's buffer for ``use`` (e.g. "gather" or "reduce") with at
        least ``numel`` elements of ``dtype``."""
        key = (use, dtype)
        buffer = self._buffers.get(key)
        if buffer is not None and buffer.tensor.numel() >= numel:
            return buffer
        if buffer is not None:
            # Collectives on any stream may still read the old buffer.
            torch.cuda.synchronize(device)
        tensor = symm_mem.empty(numel, dtype=dtype, device=device)
        # pyrefly: ignore [bad-argument-type]
        handle = symm_mem.rendezvous(tensor, self.group)
        peers = {
            other: handle.get_buffer(
                # pyrefly: ignore [bad-argument-type]
                dist.get_group_rank(self.group, self._global_ranks[other]),
                (numel,),
                dtype,
            )
            for other in self.host_ranks
        }
        buffer = SymmetricBuffer(tensor, handle, peers)
        self._buffers[key] = buffer
        return buffer


_buffers_by_mesh: dict[tuple[str, tuple[int, ...]], OwnerRowsTransportBuffers] = {}


def owner_rows_transport_buffers(
    mesh: DeviceMesh, hosts: Sequence[int]
) -> OwnerRowsTransportBuffers:
    """The mesh's transport buffers on this rank, created on first use."""
    key = (mesh.get_group().group_name, tuple(hosts))
    buffers = _buffers_by_mesh.get(key)
    if buffers is None:
        buffers = _buffers_by_mesh[key] = OwnerRowsTransportBuffers(mesh, hosts)
    return buffers


_TRANSPORT_ENV = "FLEX_SHARD_OWNER_ROWS_TRANSPORT"
_MAX_CROSSING_ENV = "FLEX_SHARD_OWNER_ROWS_MAX_CROSSING"
# Measured at 64 ranks on 8 H100 hosts: chunks beat the grouped broadcasts and
# reduces with 17%, 19% and 57% of a bucket crossing hosts in the shuffle, and
# lost with 83% (DeepSeek V3 buckets; PLAN_FLEXSHARD_TWO_LEVEL_OWNER_TRANSPORT.md).
_DEFAULT_MAX_CROSSING = 0.7


def owner_rows_transport_plan(
    mesh: DeviceMesh, row_numels: Sequence[int], device: torch.device
) -> tuple[OwnerRowsTransportPlan, tuple[int, ...]] | None:
    """The transport plan and the mesh ranks' hosts for a bucket's owner rows,
    or None where the grouped broadcasts and reduces take them as they are:
    - under torch.compile;
    - with equal rows, which already take one native collective;
    - off CUDA, or on a mesh with no host holding two of its ranks, where
      symmetric memory has nothing to share;
    - by default, when more than ``FLEX_SHARD_OWNER_ROWS_MAX_CROSSING`` (0.7)
      of the bucket would cross hosts in the shuffle. That happens when whole
      matrices are much larger than a host's share of the bucket: they spill
      into other hosts' chunks and push those hosts' rows out too.
    ``FLEX_SHARD_OWNER_ROWS_TRANSPORT`` is ``auto`` by default; ``chunks``
    takes the chunks wherever they can run, ``grouped`` never.
    """
    choice = os.environ.get(_TRANSPORT_ENV, "auto")
    if choice not in ("auto", "chunks", "grouped"):
        raise ValueError(
            f"{_TRANSPORT_ENV} must be auto, chunks or grouped, got {choice!r}."
        )
    if (
        choice == "grouped"
        or torch.compiler.is_compiling()
        or len(set(row_numels)) == 1
        or device.type != "cuda"
        or mesh.ndim != 1
    ):
        return None
    hosts = mesh_rank_hosts(mesh.mesh.flatten().tolist())
    if max(Counter(hosts).values()) < 2:
        return None
    plan = plan_owner_rows_transport(tuple(row_numels), hosts)
    max_crossing = float(os.environ.get(_MAX_CROSSING_ENV, _DEFAULT_MAX_CROSSING))
    if choice == "auto" and sum(move.numel for move in plan.host_moves) > (
        max_crossing * sum(row_numels)
    ):
        return None
    return plan, hosts


def _subtract(
    start: int, numel: int, holes: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    out, position = [], start
    for hole_start, hole_numel in sorted(holes):
        if hole_start > position:
            out.append((position, hole_start - position))
        position = max(position, hole_start + hole_numel)
    if position < start + numel:
        out.append((position, start + numel - position))
    return out


def _host_ops(
    plan: OwnerRowsTransportPlan,
    rank: int,
    buffer: torch.Tensor,
    pg: dist.ProcessGroup,
    *,
    reverse: bool,
) -> list[dist.P2POp]:
    """This rank's sends and receives between hosts, buffer to buffer."""
    ops = []
    for move in plan.host_moves:
        src, dst = (move.dst, move.src) if reverse else (move.src, move.dst)
        for peer_op, me, peer in ((dist.isend, src, dst), (dist.irecv, dst, src)):
            if me == rank:
                offset = plan.buffer_offset(rank, move.start)
                ops.append(
                    dist.P2POp(
                        peer_op,
                        buffer[offset : offset + move.numel],
                        group=pg,
                        group_peer=peer,
                    )
                )
    return ops


def _run_host_ops(ops: list[dist.P2POp]) -> None:
    """One NCCL group; the current stream then waits for it."""
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()


def _row_ranges(
    plan: OwnerRowsTransportPlan, rank: int, row_start: int, row_numel: int
) -> list[tuple[int, int, int]]:
    """Each range of this rank's row as ``(start, numel, holder)``: the rank
    whose buffer holds it in transit. That's the rank whose chunk holds it, a
    relay, or this rank, for its own chunk's part and for slices it relays."""
    row_end = row_start + row_numel
    ranges = [
        (move.start, move.numel, move.dst)
        for move in plan.owner_moves
        if move.src == rank
    ]
    ranges += [
        (move.start, move.numel, rank)
        for move in plan.host_moves
        if move.src == rank and row_start <= move.start < row_end
    ]
    ranges += [
        (start, numel, rank)
        for start, numel in _subtract(
            row_start, row_numel, [(start, numel) for start, numel, _ in ranges]
        )
    ]
    return ranges


@dataclass
class _GatherViews:
    """A bucket layout's gather on this rank, built once (see
    gather_owner_rows): the owner's writes, the sends and receives between
    hosts, the relays' writes, and the all-gather's input."""

    sources_key: tuple[Any, ...]
    writes: tuple[list[torch.Tensor], list[torch.Tensor]]
    host_ops: list[dist.P2POp]
    relays: tuple[list[torch.Tensor], list[torch.Tensor]]
    relay_barrier: bool
    chunk: torch.Tensor


def _gather_views(
    plan: OwnerRowsTransportPlan,
    hosts: Sequence[int],
    buffer: SymmetricBuffer,
    pg: dist.ProcessGroup,
    rank: int,
    row_start: int,
    row_numel: int,
    sources: Sequence[tuple[torch.Tensor, int]],
    sources_key: tuple[Any, ...],
) -> _GatherViews:
    writes: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for start, numel, holder in _row_ranges(plan, rank, row_start, row_numel):
        base = plan.buffer_offset(holder, start)
        for tensor, tensor_start in sources:
            lo = max(start, tensor_start)
            hi = min(start + numel, tensor_start + tensor.numel())
            if lo < hi:
                writes[0].append(
                    buffer.peers[holder][base + lo - start : base + hi - start]
                )
                writes[1].append(
                    tensor.reshape(-1)[lo - tensor_start : hi - tensor_start]
                )
    relays: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for move in plan.relay_moves:
        if move.src == rank:
            dst = plan.buffer_offset(move.dst, move.start)
            src = plan.buffer_offset(rank, move.start)
            relays[0].append(buffer.peers[move.dst][dst : dst + move.numel])
            relays[1].append(buffer.tensor[src : src + move.numel])
    return _GatherViews(
        sources_key=sources_key,
        writes=writes,
        host_ops=_host_ops(plan, rank, buffer.tensor, pg, reverse=False),
        relays=relays,
        relay_barrier=any(hosts[move.dst] == hosts[rank] for move in plan.relay_moves),
        chunk=buffer.tensor[: plan.chunk_numel],
    )


def gather_owner_rows(
    plan: OwnerRowsTransportPlan,
    hosts: Sequence[int],
    buffers: OwnerRowsTransportBuffers,
    *,
    rank: int,
    row_start: int,
    row_numel: int,
    sources: Sequence[tuple[torch.Tensor, int]],
    gathered: torch.Tensor,
    pg: dist.ProcessGroup,
    async_op: bool = False,
) -> dist.Work | None:
    """All-gather owner rows into ``gathered`` (``plan.gathered_numel`` long).

    This rank's row is ``[row_start, row_start + row_numel)``; ``sources`` are
    its params with their starts in the gathered buffer, in their storage.
    Each owner writes its row's slices straight from that storage into the
    buffers of the ranks whose chunks hold them, or of relays for other
    hosts, casting to ``gathered``'s dtype on the way; its own chunk's part
    goes into its own buffer. Then NCCL carries the relayed slices between
    hosts, relays write them into their holders' buffers, and NCCL's
    all-gather reads each rank's chunk. Positions no source covers are
    padding and stay unwritten.

    The views behind all this are built once per bucket layout and kept
    with the buffer; they're rebuilt when the params' storage moves."""
    buffer = buffers.get("gather", gathered.dtype, plan.buffer_numel, gathered.device)
    buffer.wait_readers()
    sources_key = tuple(
        (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.numel(),
            tensor.dtype,
            start,
        )
        for tensor, start in sources
    )
    key = ("gather", id(plan), rank)
    views = buffer.views.get(key)
    if views is None or views.sources_key != sources_key:
        views = buffer.views[key] = _gather_views(
            plan, hosts, buffer, pg, rank, row_start, row_numel, sources, sources_key
        )
    # Neighbors are done with what their buffers held, then the writes.
    buffer.barrier()
    if views.writes[0]:
        foreach_copy_(*views.writes)
    buffer.barrier()
    _run_host_ops(views.host_ops)
    if views.relays[0]:
        foreach_copy_(*views.relays)
    if views.relay_barrier:
        buffer.barrier()
    work = dist.all_gather_into_tensor(
        gathered, views.chunk, group=pg, async_op=async_op
    )
    if work is not None:
        buffer.readers.append(work)
    return work


@dataclass
class _ReduceViews:
    """A bucket layout's reduce on this rank, built once (see
    reduce_owner_rows): the relays' reads, the sends and receives between
    hosts, and the owner's reads as ``(offset in its row, numel, source)``."""

    relays: tuple[list[torch.Tensor], list[torch.Tensor]]
    host_ops: list[dist.P2POp]
    host_barrier: bool
    reads: list[tuple[int, int, torch.Tensor]]
    chunk: torch.Tensor


def _reduce_views(
    plan: OwnerRowsTransportPlan,
    hosts: Sequence[int],
    buffer: SymmetricBuffer,
    pg: dist.ProcessGroup,
    rank: int,
    row_start: int,
    row_numel: int,
) -> _ReduceViews:
    relays: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for move in plan.relay_moves:
        if move.src == rank:
            dst = plan.buffer_offset(rank, move.start)
            src = plan.buffer_offset(move.dst, move.start)
            relays[0].append(buffer.tensor[dst : dst + move.numel])
            relays[1].append(buffer.peers[move.dst][src : src + move.numel])
    reads = []
    for start, numel, holder in _row_ranges(plan, rank, row_start, row_numel):
        offset = plan.buffer_offset(holder, start)
        reads.append(
            (start - row_start, numel, buffer.peers[holder][offset : offset + numel])
        )
    return _ReduceViews(
        relays=relays,
        host_ops=_host_ops(plan, rank, buffer.tensor, pg, reverse=True),
        host_barrier=any(hosts[move.src] == hosts[rank] for move in plan.host_moves),
        reads=reads,
        chunk=buffer.tensor[: plan.chunk_numel],
    )


def reduce_owner_rows(
    plan: OwnerRowsTransportPlan,
    hosts: Sequence[int],
    buffers: OwnerRowsTransportBuffers,
    *,
    rank: int,
    row_start: int,
    send: torch.Tensor,
    recv: torch.Tensor,
    reduction: Any,
    pg: dist.ProcessGroup,
) -> None:
    """Reduce-scatter the gradient bucket ``send`` (``plan.gathered_numel``
    long, rows back to back) and bring this rank's row back into ``recv``.

    NCCL's reduce-scatter writes each rank's chunk into its buffer. Relays
    read the slices bound for other hosts from their holders, NCCL carries
    them back, and each owner reads its row's slices from the buffers of the
    ranks holding them, casting to ``recv``'s dtype on the way. A last
    barrier keeps anyone from overwriting a buffer still being read. As in
    ``gather_owner_rows``, the views are built once per bucket layout."""
    buffer = buffers.get("reduce", send.dtype, plan.buffer_numel, send.device)
    buffer.wait_readers()
    key = ("reduce", id(plan), rank)
    views = buffer.views.get(key)
    if views is None:
        views = buffer.views[key] = _reduce_views(
            plan, hosts, buffer, pg, rank, row_start, recv.numel()
        )
    reduce_scatter_grads(views.chunk, send, reduction, pg)
    buffer.barrier()
    if views.relays[0]:
        foreach_copy_(*views.relays)
    _run_host_ops(views.host_ops)
    if views.host_barrier:
        buffer.barrier()
    if views.reads:
        foreach_copy_(
            [recv[offset : offset + numel] for offset, numel, _ in views.reads],
            [source for _, _, source in views.reads],
        )
    buffer.barrier()
