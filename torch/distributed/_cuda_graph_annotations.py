"""Annotate c10d collectives with their process group.

In eager mode Kineto copies the ``record_param_comms`` metadata of a collective
onto the kernels it launched. A replayed graph has no CPU op to copy from, and
Cuspy does not see ``record_param_comms`` at all, so in both cases the NCCL kernels
carry no process group. ``torch.cuda.graph(..., enable_annotations=True)`` installs
these hooks for the length of the capture, tagging the collective kernels through
:func:`torch.cuda.graph_annotations.mark_kernels`; a Cuspy profiling session
installs them while it records, tagging eager kernels through its observer.

Collectives issued inside ``_coalescing_manager`` or a batched ``batch_isend_irecv``
are not annotated: their kernels launch when the group ends, outside any hook.
Collectives on CPU tensors are skipped, since they launch no kernels.
"""

from __future__ import annotations

import functools
import itertools
import logging
import threading
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import HookOpName, PostHookArgs, PreHookArgs
from torch.cuda._graph_annotations import mark_kernels


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    _Annotate = Callable[[dict[str, Any]], AbstractContextManager[None]]


logger = logging.getLogger(__name__)

# c10d runs hooks in ascending id order. Mirrored ids open these scopes after and
# close them before those of any hook registered with a smaller id.
_HOOK_ID_BASE = 1 << 62
_hook_offsets = itertools.count()

# Names the NCCL backend passes to record_param_comms.
_COLLECTIVE_NAMES = {
    HookOpName.SEND: "send",
    HookOpName.RECV: "recv",
    HookOpName.BROADCAST: "broadcast",
    HookOpName.ALLREDUCE: "allreduce",
    HookOpName.REDUCE: "reduce",
    HookOpName.ALLGATHER: "all_gather",
    HookOpName.REDUCE_SCATTER: "reduce_scatter",
    HookOpName.ALLTOALL: "all_to_all",
    HookOpName.BARRIER: "barrier",
    HookOpName.SCATTER: "scatter",
    HookOpName.GATHER: "gather",
    HookOpName.ALLREDUCE_COALESCED: "allreduce_coalesced",
    HookOpName.ALLGATHER_BASE: "_allgather_base",
    HookOpName.ALLGATHER_INTO_TENSOR_COALESCED: "allgather_into_tensor_coalesced",
    HookOpName.REDUCE_SCATTER_BASE: "_reduce_scatter_base",
    HookOpName.REDUCE_SCATTER_TENSOR_COALESCED: "reduce_scatter_tensor_coalesced",
    HookOpName.ALLTOALL_BASE: "all_to_allv",
}
_IN_PLACE_OPS = {
    HookOpName.BROADCAST,
    HookOpName.ALLREDUCE,
    HookOpName.REDUCE,
    HookOpName.ALLREDUCE_COALESCED,
}
# record_param_comms truncates "Process Group Ranks" past this many ranks.
_RANKS_TRUNCATE_LENGTH = 30


def _format_ranks(ranks: list[int]) -> str:
    if len(ranks) > _RANKS_TRUNCATE_LENGTH:
        head = ", ".join(map(str, ranks[: _RANKS_TRUNCATE_LENGTH - 1]))
        return f"[{head}, ..., {ranks[-1]}]"
    return f"[{', '.join(map(str, ranks))}]"


@functools.cache
def _dtype_name(dtype: torch.dtype) -> str:
    # c10::toString(ScalarType), which record_param_comms writes ("Float",
    # "BFloat16"); Tensor.type() is built from the same name.
    return torch.empty(0, dtype=dtype).type().removeprefix("torch.")[: -len("Tensor")]


class _GroupFields:
    """The fields of a group's collectives that don't change per call."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.ranks = dist.get_process_group_ranks(group)
        self.rank = group.rank()
        static: dict[str, Any] = {
            "Group size": len(self.ranks),
            "Process Group Name": group.group_name,
        }
        if group.group_desc:
            static["Process Group Description"] = group.group_desc
        static["Process Group Ranks"] = _format_ranks(self.ranks)
        self.static = static


def collective_metadata(
    group: dist.ProcessGroup,
    args: PreHookArgs,
    fields: _GroupFields | None = None,
) -> dict[str, Any]:
    """A subset of the ``record_param_comms`` fields Kineto attaches to a
    collective's kernels, in the same format. Split sizes, ``Comms Id``, the
    p2p ``Seq`` and the global rank start/stride are not available from the
    hook, and ``Seq`` can be off when threads issue collectives on one group
    concurrently."""
    if fields is None:
        fields = _GroupFields(group)
    name = _COLLECTIVE_NAMES.get(args.name, args.name.name.lower())
    inputs = args.input_tensors
    outputs = inputs if args.name in _IN_PLACE_OPS else args.output_tensors
    ranks = fields.ranks
    metadata: dict[str, Any] = {
        "Collective name": name,
        "In msg nelems": sum(t.numel() for t in inputs),
        "Out msg nelems": sum(t.numel() for t in outputs),
        **fields.static,
        "Is asynchronized op": args.async_op,
    }
    if inputs or outputs:
        metadata["dtype"] = _dtype_name((inputs or outputs)[0].dtype)
    if args.name in (HookOpName.SEND, HookOpName.RECV):
        # P2P ops advance a separate counter the group does not expose, so no Seq.
        # Like record_param_comms, "Rank" is the peer's group rank here.
        metadata["Rank"] = args.root
        if 0 <= args.root < len(ranks):
            key = "Dst Rank" if args.name == HookOpName.SEND else "Src Rank"
            metadata[key] = ranks[args.root]
    else:
        metadata["Rank"] = fields.rank
        try:
            # The hook fires before the backend bumps its counter.
            metadata["Seq"] = (
                group._get_sequence_number_for_group()  # pyrefly: ignore[missing-attribute]
                + 1
            )
        except RuntimeError:
            pass
    return metadata


def _launches_kernels(args: PreHookArgs) -> bool:
    tensors = args.input_tensors or args.output_tensors
    return bool(tensors) and tensors[0].is_cuda


def _mark_kernels(metadata: dict[str, Any]) -> AbstractContextManager[None]:
    # Backward attribution would tag whatever autograd node launched the
    # collective, not the collective itself.
    return mark_kernels(metadata, backward=False)


class _GroupHooks:
    def __init__(
        self, group: dist.ProcessGroup, annotate: _Annotate, *, eager: bool
    ) -> None:
        self._group = group
        self._annotate = annotate
        self._eager = eager
        self._fields = _GroupFields(group)
        offset = next(_hook_offsets)
        self._pre_id = _HOOK_ID_BASE + offset
        self._post_id = -_HOOK_ID_BASE - offset
        self._lock = threading.Lock()
        self._closed = False
        # op_id -> (thread that entered the scope, scope)
        self._scopes: dict[int, tuple[int, AbstractContextManager[None]]] = {}
        group.register_pre_hook(self._pre_id, self._pre)
        try:
            group.register_post_hook(self._post_id, self._post)
        except BaseException:
            group.unregister_pre_hook(self._pre_id)
            raise

    # A raising hook fails the collective, so annotation errors are only logged.
    def _pre(self, args: PreHookArgs) -> None:
        # The hooks fire on every thread: without eager, skip collectives other
        # threads issue during a capture, or they would enter the capture's
        # module-global annotation stack.
        if not _launches_kernels(args) or (
            not self._eager and not torch.cuda.is_current_stream_capturing()
        ):
            return
        try:
            scope = self._annotate(collective_metadata(self._group, args, self._fields))
            scope.__enter__()
        except Exception:
            logger.exception("Failed to annotate %s", args.name)
            return
        with self._lock:
            closed = self._closed
            if not closed:
                self._scopes[args.op_id] = (threading.get_ident(), scope)
        if closed:
            # Fired from a hook snapshot taken before close(); the post hook may
            # already be gone.
            self._exit(scope)

    def _post(self, args: PostHookArgs) -> None:
        with self._lock:
            entry = self._scopes.pop(args.op_id, None)
            unregister = self._closed and not self._scopes
        if entry is not None:
            self._exit(entry[1])
        if unregister:
            self._group.unregister_post_hook(self._post_id)

    @staticmethod
    def _exit(scope: AbstractContextManager[None]) -> None:
        try:
            scope.__exit__(None, None, None)
        except Exception:
            logger.exception("Failed to close collective annotation")

    def close(self) -> None:
        """Stops annotating new collectives. Scopes this thread opened (left
        over from a backend that raised) are closed now; a collective still
        between its hooks on another thread closes its own scope, then the post
        hook unregisters."""
        self._group.unregister_pre_hook(self._pre_id)
        tid = threading.get_ident()
        with self._lock:
            self._closed = True
            mine = [op for op, (owner, _) in self._scopes.items() if owner == tid]
            stale = [self._scopes.pop(op)[1] for op in mine]
            unregister = not self._scopes
        for scope in reversed(stale):
            self._exit(scope)
        if unregister:
            self._group.unregister_post_hook(self._post_id)


class CollectiveAnnotations:
    """Tags collectives on every process group that exists when it is created,
    until :meth:`close`. Groups created afterwards are not hooked.

    ``annotate`` maps a collective's metadata to the scope its kernels launch in;
    it defaults to :func:`~torch.cuda.graph_annotations.mark_kernels`. Unless
    ``eager``, only collectives issued from a capturing stream are tagged.
    """

    def __init__(
        self, annotate: _Annotate = _mark_kernels, *, eager: bool = False
    ) -> None:
        self._hooks: list[_GroupHooks] = []
        from torch.distributed.distributed_c10d import _world

        try:
            for group in list(_world.pg_map):
                self._hooks.append(_GroupHooks(group, annotate, eager=eager))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Never raises: ``torch.cuda.graph`` calls it before ending the capture."""
        while self._hooks:
            try:
                self._hooks.pop().close()
            except Exception:
                logger.exception("Failed to remove collective annotation hooks")
