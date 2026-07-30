"""Prototype virtual/fake ProcessGroups implemented in Python.

This module provides two building blocks for running a large *logical*
distributed topology inside a much smaller *physical* world:

- ``LocalProcessGroup``: a reusable, all-Python fake ProcessGroup. It is
  constructed standalone (no ``init_process_group``, no global ``_world``
  state), has deterministic local semantics for every collective, and routes
  every collective through a single normalized hook, ``run_collective``.

- ``VirtualProcessGroup``: a ``LocalProcessGroup`` that additionally mirrors
  every logical collective onto a real physical ProcessGroup (e.g. NCCL or
  gloo) and returns the *physical* Work object so that ``wait()``
  synchronizes with the real communication at the actual consumer. Three
  fidelity levels via ``output_mode``: ``"projected"`` runs the physical
  collective directly on views of the application tensors (true
  producer -> collective -> consumer dependencies, CUDA-graph faithful),
  ``"local_fake"`` fills logical outputs deterministically while mirroring
  on scratch, and ``"scratch"`` exercises communication structure only.

- ``install_virtual_world``: installs a VirtualProcessGroup as the default
  process group so unmodified applications (``dist.get_rank``,
  ``dist.new_group``, ``DeviceMesh``) observe the logical world, with
  logical subgroups optionally backed by distinct physical split
  communicators (``mirror_split="split"``).

Both eager collectives (``pg.allreduce(...)``, ``dist.all_reduce(t, group=pg)``)
and functional collectives (``_c10d_functional`` ops, by group name or PG
object, eager or under torch.compile) dispatch through the same hook, because
the interception happens at the ProcessGroup virtual-method boundary via the
PyProcessGroup trampoline.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (  # pyrefly: ignore[missing-module-attribute]
    _create_work_from_future,
    _register_process_group,
    _unregister_process_group,
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    ReduceScatterOptions,
)
from torch.futures import Future


__all__ = [
    "Collective",
    "LocalProcessGroup",
    "ProjectionError",
    "VirtualProcessGroup",
    "create_physical_group",
    "install_virtual_world",
    "uninstall_virtual_world",
]


_local_pg_count = 0


def _completed_work(result: Any) -> dist._Work:
    fut: Future = Future()
    fut.set_result(result)
    return _create_work_from_future(fut)


@dataclass
class Collective:
    """Normalized description of one collective call.

    Every ProcessGroup entry point (eager and functional) is translated into
    one of these before being passed to ``run_collective``. ``inputs`` and
    ``outputs`` are flat tensor lists; ops with nested list signatures (e.g.
    ``allgather``) keep their original structure in ``opts_args``.
    """

    op: str
    inputs: list[torch.Tensor] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    reduce_op: dist.ReduceOp | None = None
    root: int | None = None
    peer: int | None = None
    tag: int | None = None
    input_splits: list[int] | None = None
    output_splits: list[int] | None = None
    # Original (unnormalized) positional args, for hooks that need full detail
    opts_args: tuple = ()


class LocalProcessGroup(dist.ProcessGroup):
    """All-Python fake ProcessGroup with a logical rank/world size.

    Standalone: does not require or touch ``init_process_group`` state. The
    logical rank/world size are independent of any physical transport. All
    collectives complete immediately with deterministic local semantics
    (documented per-op in ``_default_semantics``); reductions degenerate to
    identity because only this rank's data is available locally.

    Subclasses customize behavior by overriding a single hook::

        def run_collective(self, coll: Collective) -> Optional[dist._Work]

    Returning ``None`` means "local semantics were applied; complete
    immediately". Returning a Work defers completion to that Work (this is
    how ``VirtualProcessGroup`` returns real physical Work objects).
    """

    def __init__(self, rank: int, world_size: int, group_name: str | None = None):
        if not (0 <= rank < world_size):
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        # pyrefly: ignore[missing-argument, bad-argument-type]
        super().__init__(rank, world_size)
        global _local_pg_count
        if group_name is None:
            group_name = f"local_pg_{_local_pg_count}"
            _local_pg_count += 1
        # The base C++ getGroupName() requires a registered backend, which a
        # pure-Python PG does not have, so keep the name Python-side.
        self._name = group_name
        self._desc = "undefined"
        self._registered = False
        self.register()

    # -- registration in the C++ group registry (funcol name resolution) --

    def register(self) -> "LocalProcessGroup":
        if not self._registered:
            # pyrefly: ignore[bad-argument-type]
            _register_process_group(self._name, self)
            self._registered = True
        return self

    def unregister(self) -> None:
        if self._registered:
            # pyrefly: ignore[bad-argument-type]
            _unregister_process_group(self._name)
            self._registered = False

    # -- metadata overrides (trampoline dispatches these to Python) --

    def getBackendName(self) -> str:
        return "local-py"

    def getGroupName(self) -> str:
        return self._name

    def setGroupName(self, name: str) -> None:
        self._name = name

    def getGroupDesc(self) -> str:
        return self._desc

    def setGroupDesc(self, desc: str) -> None:
        self._desc = desc

    # -- the single normalized hook --

    def run_collective(self, coll: Collective) -> dist._Work | None:
        return None

    def _dispatch(self, coll: Collective) -> dist._Work:
        if self._use_local_semantics(coll):
            self._default_semantics(coll)
        work = self.run_collective(coll)
        if work is None:
            work = _completed_work(coll.outputs if coll.outputs else coll.inputs)
        return work

    def _use_local_semantics(self, coll: Collective) -> bool:
        """Whether ``_dispatch`` fills outputs with deterministic local data.

        Subclasses that produce the logical output by other means (e.g. a
        projected physical collective) return False so no extra fill kernels
        are issued, which matters under CUDA graph capture.
        """
        return True

    # -- deterministic local semantics --

    def _default_semantics(self, coll: Collective) -> None:
        """Apply deterministic local data semantics for ``coll`` in place.

        Reductions are identity (only local data available); gather-like ops
        replicate the local input; scatter-like ops select this rank's chunk.
        Mirrors the C++ FakeProcessGroup semantics so outputs are never left
        uninitialized. Copies run below autograd so requires_grad inputs work.
        """
        with torch.no_grad():
            self._apply_semantics(coll)

    def _apply_semantics(self, coll: Collective) -> None:
        op = coll.op
        if op in (
            "allreduce",
            "allreduce_coalesced",
            "reduce",
            "broadcast",
            "barrier",
            "send",
            "recv",
            "recv_anysource",
        ):
            return  # in-place identity or no local effect
        if op == "allgather":
            output_lists, input_tensors = coll.opts_args[0], coll.opts_args[1]
            for outs, inp in zip(output_lists, input_tensors):
                for o in outs:
                    o.detach().copy_(inp)
        elif op == "all_gather_single":
            out, inp = coll.outputs[0], coll.inputs[0]
            for chunk in out.detach().chunk(self.size()):
                chunk.copy_(inp)
        elif op == "all_gather_single_coalesced":
            for out, inp in zip(coll.outputs, coll.inputs):
                for chunk in out.detach().chunk(self.size()):
                    chunk.copy_(inp)
        elif op == "gather":
            if coll.root == self.rank():
                output_lists, input_tensors = coll.opts_args[0], coll.opts_args[1]
                for outs, inp in zip(output_lists, input_tensors):
                    for o in outs:
                        o.detach().copy_(inp)
        elif op == "scatter":
            if coll.root == self.rank():
                input_lists = coll.opts_args[1]
                for out, ins in zip(coll.outputs, input_lists):
                    out.detach().copy_(ins[self.rank()])
        elif op == "reduce_scatter":
            input_lists = coll.opts_args[1]
            for out, ins in zip(coll.outputs, input_lists):
                out.detach().copy_(ins[self.rank()])
        elif op in ("reduce_scatter_single", "reduce_scatter_single_coalesced"):
            for out, inp in zip(coll.outputs, coll.inputs):
                out.detach().copy_(inp.chunk(self.size())[self.rank()])
        elif op == "alltoall":
            for out, inp in zip(coll.outputs, coll.inputs):
                out.detach().copy_(inp)
        elif op == "all_to_all_single":
            out, inp = coll.outputs[0], coll.inputs[0]
            if not coll.input_splits and not coll.output_splits:
                out.detach().copy_(inp)
            else:
                # Uneven splits: this rank's contribution to itself is the
                # only locally-knowable piece; fill the rest by repeating it.
                in_splits = (
                    coll.input_splits or [inp.size(0) // self.size()] * self.size()
                )
                start = sum(in_splits[: self.rank()])
                chunk = inp[start : start + in_splits[self.rank()]]
                out_splits = (
                    coll.output_splits or [out.size(0) // self.size()] * self.size()
                )
                off = 0
                for s in out_splits:
                    n = min(s, chunk.size(0))
                    if n > 0:
                        out.detach()[off : off + n].copy_(chunk[:n])
                    off += s
        else:
            raise NotImplementedError(f"LocalProcessGroup: unhandled op {op}")

    # -- ProcessGroup virtual methods, normalized into Collective --

    # pyrefly: ignore[bad-override]
    def broadcast(self, tensors: list[torch.Tensor], opts: Any = None) -> dist._Work:
        return self._dispatch(
            Collective(
                "broadcast",
                inputs=list(tensors),
                outputs=list(tensors),
                root=opts.rootRank,
                opts_args=(tensors, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def allreduce(self, tensors: list[torch.Tensor], opts: Any = None) -> dist._Work:
        return self._dispatch(
            Collective(
                "allreduce",
                inputs=list(tensors),
                outputs=list(tensors),
                reduce_op=opts.reduceOp,
                opts_args=(tensors, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def allreduce_coalesced(
        self, tensors: list[torch.Tensor], opts: Any = None
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "allreduce_coalesced",
                inputs=list(tensors),
                outputs=list(tensors),
                reduce_op=opts.reduceOp,
                opts_args=(tensors, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def reduce(self, tensors: list[torch.Tensor], opts: Any = None) -> dist._Work:
        return self._dispatch(
            Collective(
                "reduce",
                inputs=list(tensors),
                outputs=list(tensors),
                reduce_op=opts.reduceOp,
                root=opts.rootRank,
                opts_args=(tensors, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def allgather(
        self,
        output_lists: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: Any = None,
    ) -> dist._Work:
        flat_out = [o for outs in output_lists for o in outs]
        return self._dispatch(
            Collective(
                "allgather",
                inputs=list(input_tensors),
                outputs=flat_out,
                opts_args=(output_lists, input_tensors, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, opts: Any = None
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "all_gather_single",
                inputs=[input],
                outputs=[output],
                opts_args=(output, input, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def all_gather_single_coalesced(
        self, outputs: list[torch.Tensor], inputs: list[torch.Tensor], opts: Any = None
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "all_gather_single_coalesced",
                inputs=list(inputs),
                outputs=list(outputs),
                opts_args=(outputs, inputs, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def gather(
        self,
        output_lists: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: Any = None,
    ) -> dist._Work:
        flat_out = [o for outs in output_lists for o in outs]
        return self._dispatch(
            Collective(
                "gather",
                inputs=list(input_tensors),
                outputs=flat_out,
                root=opts.rootRank,
                opts_args=(output_lists, input_tensors, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_lists: list[list[torch.Tensor]],
        opts: Any = None,
    ) -> dist._Work:
        flat_in = [i for ins in input_lists for i in ins]
        return self._dispatch(
            Collective(
                "scatter",
                inputs=flat_in,
                outputs=list(output_tensors),
                root=opts.rootRank,
                opts_args=(output_tensors, input_lists, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_lists: list[list[torch.Tensor]],
        opts: Any = None,
    ) -> dist._Work:
        flat_in = [i for ins in input_lists for i in ins]
        return self._dispatch(
            Collective(
                "reduce_scatter",
                inputs=flat_in,
                outputs=list(output_tensors),
                reduce_op=opts.reduceOp,
                opts_args=(output_tensors, input_lists, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def reduce_scatter_single(
        self, output: torch.Tensor, input: torch.Tensor, opts: Any = None
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "reduce_scatter_single",
                inputs=[input],
                outputs=[output],
                reduce_op=opts.reduceOp,
                opts_args=(output, input, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def reduce_scatter_single_coalesced(
        self, outputs: list[torch.Tensor], inputs: list[torch.Tensor], opts: Any = None
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "reduce_scatter_single_coalesced",
                inputs=list(inputs),
                outputs=list(outputs),
                reduce_op=opts.reduceOp,
                opts_args=(outputs, inputs, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def alltoall(
        self, outputs: list[torch.Tensor], inputs: list[torch.Tensor], opts: Any = None
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "alltoall",
                inputs=list(inputs),
                outputs=list(outputs),
                opts_args=(outputs, inputs, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        opts: Any = None,
    ) -> dist._Work:
        return self._dispatch(
            Collective(
                "all_to_all_single",
                inputs=[input],
                outputs=[output],
                input_splits=list(input_splits),
                output_splits=list(output_splits),
                opts_args=(output, input, output_splits, input_splits, opts),
            )
        )

    # pyrefly: ignore[bad-override]
    def barrier(self, opts: Any = None) -> dist._Work:
        return self._dispatch(Collective("barrier", opts_args=(opts,)))

    # pyrefly: ignore[bad-override]
    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> dist._Work:
        return self._dispatch(
            Collective(
                "send",
                inputs=list(tensors),
                peer=dst_rank,
                tag=tag,
                opts_args=(tensors, dst_rank, tag),
            )
        )

    # pyrefly: ignore[bad-override]
    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> dist._Work:
        return self._dispatch(
            Collective(
                "recv",
                outputs=list(tensors),
                peer=src_rank,
                tag=tag,
                opts_args=(tensors, src_rank, tag),
            )
        )

    # pyrefly: ignore[bad-override]
    def recvAnysource(self, tensors: list[torch.Tensor], tag: int) -> dist._Work:
        return self._dispatch(
            Collective(
                "recv_anysource",
                outputs=list(tensors),
                tag=tag,
                opts_args=(tensors, tag),
            )
        )

    # -- logical subgroup creation --

    def split_local(
        self, ranks: list[int], group_name: str | None = None
    ) -> Optional["LocalProcessGroup"]:
        """Create a logical subgroup of this group.

        ``ranks`` are ranks of *this* group (parent-local, not global). The
        new group's rank is this rank's position in ``ranks``. Returns None
        if this rank is not a member. This is standalone splitting: it does
        not require the default process group or ``_world`` registration,
        unlike ``dist.split_group``.
        """
        for r in ranks:
            if not (0 <= r < self.size()):
                raise ValueError(f"rank {r} out of range for world size {self.size()}")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"duplicate ranks in {ranks}")
        if self.rank() not in ranks:
            return None
        if group_name is None:
            group_name = f"{self._name}:split:{'_'.join(map(str, ranks))}"
        return type(self)._split_impl(self, ranks, group_name)

    @classmethod
    def _split_impl(
        cls, parent: "LocalProcessGroup", ranks: list[int], group_name: str
    ) -> "LocalProcessGroup":
        return LocalProcessGroup(ranks.index(parent.rank()), len(ranks), group_name)

    def splitGroup(
        self,
        ranks: list[int],
        timeout: Any = None,
        opts: Any = None,
        group_name: str | None = None,
        group_desc: str | None = None,
        devices: Any = None,
    ) -> Optional["LocalProcessGroup"]:
        pg = self.split_local(list(ranks), group_name)
        if pg is not None and group_desc is not None:
            pg._desc = group_desc
        return pg


class ProjectionError(RuntimeError):
    """Raised when ``output_mode="projected"`` cannot map a logical collective
    onto views of the application tensors. Never silently falls back to
    scratch buffers; callers that can tolerate structure-only mirroring must
    opt into ``output_mode="scratch"`` explicitly."""


class VirtualProcessGroup(LocalProcessGroup):
    """Logical ProcessGroup mirrored onto a real physical ProcessGroup.

    Presents a logical (rank, world_size) independent of the physical
    transport, while every logical collective issues a *real* collective of
    the same kind on ``physical_group`` and returns the physical Work, so
    ``wait()`` at the consumer synchronizes with real communication.

    ``output_mode`` selects how the physical collective relates to the
    application's tensors:

    - ``"projected"``: the physical collective executes directly on
      contiguous views of the logical input/output tensors, creating true
      producer-kernel -> NCCL -> consumer-kernel dependencies in the stream
      (and thus in a captured CUDA graph). The physical collective writes the
      reachable prefix of the logical output; logical-rank data beyond the
      physical world size is undefined. No local fill kernels are issued.
      Raises ProjectionError when a faithful view mapping is impossible.
    - ``"local_fake"``: deterministic local semantics fill the logical
      outputs (useful for ordinary tests); the physical collective runs on
      private scratch buffers.
    - ``"scratch"``: physical collective on private scratch buffers only;
      logical outputs are left untouched. Exercises communication structure
      but has NO tensor-data dependency on the application's producers or
      consumers.

    In ``"projected"`` mode there are no allocations at issue time (views
    only); warm up once before CUDA graph capture so the physical
    communicator is initialized. In scratch-based modes the scratch pool
    grows on first use per (op, size, dtype, device) — warm up every shape
    before capture.

    ``physical_peer_map`` (logical rank -> physical rank) enables real
    mirroring of send/recv; without it P2P is a local no-op.

    Logical splits (``split_local``/``splitGroup``) produce a
    ``VirtualProcessGroup`` sharing the same physical group by default, or a
    physically split communicator with ``mirror_split="split"`` when the
    physical backend supports splitting.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        physical_group: dist.ProcessGroup,
        group_name: str | None = None,
        mirror_split: str = "reuse",
        output_mode: str = "local_fake",
        physical_peer_map: dict[int, int] | None = None,
    ):
        if mirror_split not in ("reuse", "split"):
            raise ValueError(
                f"mirror_split must be 'reuse' or 'split', got {mirror_split!r}"
            )
        if output_mode not in ("projected", "local_fake", "scratch"):
            raise ValueError(
                f"output_mode must be 'projected', 'local_fake' or 'scratch', "
                f"got {output_mode!r}"
            )
        self.physical_group = physical_group
        self.mirror_split = mirror_split
        self.output_mode = output_mode
        self.physical_peer_map = physical_peer_map
        # keyed by (op, numel_per_rank, dtype, device); persistent so mirrored
        # collectives have stable buffer addresses (CUDA-graph friendly)
        self._scratch: dict[tuple, list[torch.Tensor]] = {}
        super().__init__(rank, world_size, group_name)

    def getBackendName(self) -> str:
        return "virtual-py"

    def _use_local_semantics(self, coll: Collective) -> bool:
        return self.output_mode == "local_fake"

    def run_collective(self, coll: Collective) -> dist._Work | None:
        if coll.op in ("barrier", "recv_anysource"):
            return None
        with torch.profiler.record_function(f"virtual_pg::{self._name}::{coll.op}"):
            if coll.op in ("send", "recv"):
                return self._mirror_p2p(coll)
            if self.output_mode == "projected":
                return self._mirror_projected(coll)
            return self._mirror_scratch(coll)

    # -- P2P mirroring --

    def _mirror_p2p(self, coll: Collective) -> dist._Work | None:
        if self.physical_peer_map is None:
            if self.output_mode == "projected":
                raise ProjectionError(
                    f"{coll.op} in projected mode requires physical_peer_map"
                )
            return None
        if coll.peer is None:
            raise ProjectionError(f"{coll.op} missing peer rank")
        phys_peer = self.physical_peer_map.get(coll.peer)
        if phys_peer is None:
            raise ProjectionError(f"logical peer {coll.peer} not in physical_peer_map")
        tensors = coll.inputs if coll.op == "send" else coll.outputs
        # P2P operates directly on the application tensors: full fidelity.
        if coll.op == "send":
            return self.physical_group.send(tensors, phys_peer, coll.tag or 0)
        return self.physical_group.recv(tensors, phys_peer, coll.tag or 0)

    # -- projected mirroring: physical collective on views of logical tensors --

    @staticmethod
    def _flat_view(t: torch.Tensor, numel: int, what: str) -> torch.Tensor:
        if not t.is_contiguous():
            raise ProjectionError(f"{what} must be contiguous for projection")
        if t.numel() < numel:
            raise ProjectionError(
                f"{what} has {t.numel()} elements, projection needs {numel}"
            )
        return t.detach().reshape(-1)[:numel]

    def _mirror_projected(self, coll: Collective) -> dist._Work:
        """Run the physical collective directly on views of the logical
        tensors, so the collective inherits the application's producer
        dependencies and its consumers depend on the collective.

        Only single-buffer collective forms project faithfully; list-form ops
        (allgather into separate output tensors, gather/scatter/alltoall
        lists) raise ProjectionError because their outputs are not one
        contiguous buffer.
        """
        pg = self.physical_group
        pw = pg.size()
        op = coll.op
        work: dist._Work | None = None
        if op in ("allreduce", "allreduce_coalesced"):
            opts = AllreduceOptions()
            if coll.reduce_op is not None:
                opts.reduceOp = coll.reduce_op
            for t in coll.inputs:
                if not t.is_contiguous():
                    raise ProjectionError("allreduce tensor must be contiguous")
                work = pg.allreduce([t.detach()], opts)
        elif op == "broadcast":
            from torch._C._distributed_c10d import BroadcastOptions

            opts = BroadcastOptions()
            # physical root: logical root's data only exists physically when
            # some physical rank holds it; fold into the physical namespace
            opts.rootRank = (coll.root or 0) % pw
            for t in coll.inputs:
                if not t.is_contiguous():
                    raise ProjectionError("broadcast tensor must be contiguous")
                work = pg.broadcast([t.detach()], opts)
        elif op in ("all_gather_single", "all_gather_single_coalesced"):
            for out, inp in zip(coll.outputs, coll.inputs):
                n = inp.numel()
                in_view = self._flat_view(inp, n, "all_gather input")
                out_view = self._flat_view(out, pw * n, "all_gather output")
                work = pg.all_gather_single(out_view, in_view, AllgatherOptions())
        elif op in ("reduce_scatter_single", "reduce_scatter_single_coalesced"):
            opts = ReduceScatterOptions()
            if coll.reduce_op is not None:
                opts.reduceOp = coll.reduce_op
            for out, inp in zip(coll.outputs, coll.inputs):
                n = out.numel()
                in_view = self._flat_view(inp, pw * n, "reduce_scatter input")
                out_view = self._flat_view(out, n, "reduce_scatter output")
                work = pg.reduce_scatter_single(out_view, in_view, opts)
        elif op == "all_to_all_single":
            if coll.input_splits or coll.output_splits:
                raise ProjectionError(
                    "all_to_all_single with uneven splits cannot be projected"
                )
            inp, out = coll.inputs[0], coll.outputs[0]
            if inp.numel() % self.size() != 0:
                raise ProjectionError(
                    "all_to_all_single input not divisible by logical world"
                )
            n = inp.numel() // self.size()
            in_view = self._flat_view(inp, pw * n, "all_to_all input")
            out_view = self._flat_view(out, pw * n, "all_to_all output")
            work = pg.all_to_all_single(out_view, in_view, [], [], AllToAllOptions())
        else:
            raise ProjectionError(
                f"{op} cannot be projected onto model buffers; use "
                f"output_mode='scratch' explicitly if communication "
                f"structure alone is sufficient"
            )
        if work is None:
            raise AssertionError(f"projection produced no work for {op}")
        return work

    # -- scratch mirroring: communication structure only --

    def _scratch_for(
        self, op: str, ref: torch.Tensor, numel_per_rank: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Persistent (input, output) scratch pair for one mirrored collective."""
        pw = self.physical_group.size()
        key = (op, numel_per_rank, ref.dtype, ref.device)
        if key not in self._scratch:
            inp = torch.empty(numel_per_rank * pw, dtype=ref.dtype, device=ref.device)
            out = torch.empty(numel_per_rank * pw, dtype=ref.dtype, device=ref.device)
            self._scratch[key] = [inp, out]
        return self._scratch[key][0], self._scratch[key][1]

    def _mirror_scratch(self, coll: Collective) -> dist._Work:
        """Issue a real collective of the same kind on private scratch
        buffers. Communication volume per physical rank approximates the
        logical per-rank volume, but the collective has no tensor dependency
        on the application's producers or consumers; only the returned Work
        links it to the logical program.
        """
        ref = coll.inputs[0] if coll.inputs else coll.outputs[0]
        pg = self.physical_group
        pw = pg.size()
        n = max(ref.numel(), 1)
        op = coll.op
        with torch.no_grad():
            if op in ("allreduce", "allreduce_coalesced", "reduce", "broadcast"):
                inp, _ = self._scratch_for("allreduce", ref, n)
                buf = inp[:n]
                buf.copy_(ref.detach().reshape(-1))
                opts = AllreduceOptions()
                if coll.reduce_op is not None and op != "broadcast":
                    opts.reduceOp = coll.reduce_op
                return pg.allreduce([buf], opts)
            if op in (
                "allgather",
                "all_gather_single",
                "all_gather_single_coalesced",
                "gather",
            ):
                inp, out = self._scratch_for("all_gather", ref, n)
                buf = inp[:n]
                buf.copy_(ref.detach().reshape(-1))
                return pg.all_gather_single(out, buf, AllgatherOptions())
            if op in (
                "reduce_scatter",
                "reduce_scatter_single",
                "reduce_scatter_single_coalesced",
                "scatter",
            ):
                inp, out = self._scratch_for("reduce_scatter", ref, n)
                inp.copy_(
                    ref.detach().reshape(-1).expand(pw, n).reshape(-1)
                    if ref.numel()
                    else inp
                )
                opts = ReduceScatterOptions()
                if coll.reduce_op is not None:
                    opts.reduceOp = coll.reduce_op
                return pg.reduce_scatter_single(out[:n], inp, opts)
            if op in ("alltoall", "all_to_all_single"):
                inp, out = self._scratch_for("all_to_all", ref, n)
                inp[:n].copy_(ref.detach().reshape(-1))
                return pg.all_to_all_single(out, inp, [], [], AllToAllOptions())
        raise NotImplementedError(f"VirtualProcessGroup: unhandled op {op}")

    @classmethod
    def _split_impl(
        cls, parent: "LocalProcessGroup", ranks: list[int], group_name: str
    ) -> "LocalProcessGroup":
        if not isinstance(parent, VirtualProcessGroup):
            raise TypeError(f"expected VirtualProcessGroup, got {type(parent)}")
        phys = parent.physical_group
        if parent.mirror_split == "split":
            new_phys = phys.split_group(
                list(range(phys.size())),
                None,
                None,
                # pyrefly: ignore[bad-argument-type]
                f"{group_name}:phys",
                None,
                None,
            )
            if new_phys is None:
                raise RuntimeError("physical split_group returned no group")
            phys = new_phys
        return VirtualProcessGroup(
            ranks.index(parent.rank()),
            len(ranks),
            phys,
            group_name,
            parent.mirror_split,
            parent.output_mode,
            parent.physical_peer_map,
        )

    def new_group(
        self,
        ranks: list[int],
        timeout: Any = None,
        backend: Any = None,
        pg_options: Any = None,
        group_name: str | None = None,
        group_desc: str | None = None,
    ) -> Optional["VirtualProcessGroup"]:
        """Hook for ``dist.new_group`` delegation when this group is the
        default world (see ``_new_group_with_tag``): creates a virtual
        logical child. ``ranks`` are logical ranks of this group. With
        ``mirror_split="split"`` each logical child receives a distinct
        physical child communicator, split deterministically from the
        physical parent with ALL physical ranks (physical-parent-local rank
        namespace; the logical namespace never leaks into the physical
        split). All physical processes must therefore call ``new_group`` in
        the same order, including for logical groups they are not members
        of, mirroring the standard c10d contract.
        """
        # Physical child creation must happen on every physical process and
        # in deterministic order, so do it before the membership early-out.
        phys = self._physical_child(group_name)
        if self.rank() not in ranks:
            return None
        child = VirtualProcessGroup(
            ranks.index(self.rank()),
            len(ranks),
            phys,
            group_name,
            self.mirror_split,
            self.output_mode,
            self.physical_peer_map,
        )
        if group_desc is not None:
            child._desc = group_desc
        # The caller (_new_group_with_tag) registers the group in the C++
        # registry via _register_pg_in_world; drop our ctor registration so
        # the name is not claimed twice.
        child.unregister()
        return child

    def _physical_child(self, group_name: str | None) -> dist.ProcessGroup:
        if self.mirror_split != "split":
            return self.physical_group
        phys = self.physical_group
        new_phys = phys.split_group(
            list(range(phys.size())),
            None,
            None,
            # pyrefly: ignore[bad-argument-type]
            f"{group_name}:phys" if group_name else None,
            None,
            None,
        )
        if new_phys is None:
            raise RuntimeError("physical split_group returned no group")
        return new_phys


def create_physical_group(
    store: dist.Store,
    rank: int,
    world_size: int,
    device: torch.device | None = None,
    group_name: str = "virtual_pg_physical",
) -> dist.ProcessGroup:
    """Create a standalone physical ProcessGroup suitable as a mirror target.

    Builds a ``ProcessGroup`` with an NCCL backend (if ``device`` is a CUDA
    device) or gloo backend, without touching the default-group machinery.
    For NCCL the communicator is eagerly connected so the group is
    ``split_group``-capable (``ncclCommSplit``) and safe to use during CUDA
    graph capture. This group lives in the *physical* rank namespace: its
    rank/world size come from the transport, not from any logical topology.
    """
    pg = dist.ProcessGroup(store, rank, world_size)
    # pyrefly: ignore[bad-argument-type]
    pg._set_group_name(group_name)
    if device is not None and device.type == "cuda":
        opts = dist.ProcessGroupNCCL.Options()
        nccl = dist.ProcessGroupNCCL(store, rank, world_size, opts)
        nccl.eager_connect_single_device(device)
        pg._register_backend(device, dist.ProcessGroup.BackendType.NCCL, nccl)
        pg._set_default_backend(dist.ProcessGroup.BackendType.NCCL)
    else:
        gloo = dist.ProcessGroupGloo(store, rank, world_size)
        pg._register_backend(
            torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO, gloo
        )
        pg._set_default_backend(dist.ProcessGroup.BackendType.GLOO)
    return pg


def install_virtual_world(
    virtual_pg: VirtualProcessGroup, store: dist.Store | None = None
) -> VirtualProcessGroup:
    """Install ``virtual_pg`` as the default process group.

    After this, an unmodified application sees the logical world:
    ``dist.get_rank()``/``get_world_size()`` report logical values,
    ``dist.new_group()`` creates virtual logical children via
    ``VirtualProcessGroup.new_group`` (each optionally backed by a distinct
    physical child communicator), and ``DeviceMesh`` builds its dimension
    groups through the same path.

    ``init_process_group`` must not already have been called. Undo with
    ``uninstall_virtual_world``.
    """
    import torch.distributed.distributed_c10d as c10d

    if c10d.is_initialized():
        raise RuntimeError(
            "install_virtual_world requires an uninitialized default group"
        )
    if store is None:
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
    # Register the backend name so BackendConfig in the new_group delegation
    # path resolves it without warning. register_backend rejects duplicates,
    # so only do it once per process.
    name = virtual_pg.getBackendName()
    if name.upper() not in c10d.Backend._plugins:
        c10d.Backend.register_backend(
            name,
            lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("virtual PGs are constructed via new_group delegation")
            ),
            devices=["cpu", "cuda"],
        )
    c10d._update_default_pg(virtual_pg)
    c10d._world.pg_map[virtual_pg] = (virtual_pg.getBackendName(), store)
    c10d._world.pg_names[virtual_pg] = virtual_pg.group_name
    c10d._world.pg_backend_config[virtual_pg] = virtual_pg.getBackendName()
    c10d._world.pg_group_ranks[virtual_pg] = {i: i for i in range(virtual_pg.size())}
    tag = f"ptd:{virtual_pg.group_name}"
    c10d._world.tags_to_pg.setdefault("", []).append(virtual_pg)
    c10d._world.tags_to_pg.setdefault(tag, []).append(virtual_pg)
    c10d._world.pg_to_tag[virtual_pg] = tag
    return virtual_pg


def uninstall_virtual_world() -> None:
    """Remove a virtual default world installed by ``install_virtual_world``.

    Uses ``destroy_process_group`` teardown semantics for the ``_world``
    dictionaries but leaves the physical group alive (the caller owns it).
    """
    import torch.distributed.distributed_c10d as c10d

    default_pg = c10d._world.default_pg
    if default_pg is not None:
        for d in (
            c10d._world.pg_map,
            c10d._world.pg_names,
            c10d._world.pg_backend_config,
            c10d._world.pg_group_ranks,
            c10d._world.pg_to_tag,
        ):
            d.pop(default_pg, None)
        for pgs in c10d._world.tags_to_pg.values():
            while default_pg in pgs:
                pgs.remove(default_pg)
    c10d._update_default_pg(None)
