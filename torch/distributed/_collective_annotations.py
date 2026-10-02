"""Annotate c10d collectives with their process group.

In eager mode Kineto copies the ``record_param_comms`` metadata of a collective
onto the kernels it launched. A replayed CUDA graph has no CPU op to copy from, and
Cuspy does not see ``record_param_comms`` at all, so in both cases the NCCL kernels
carry no process group. Every process group gets gated c10d hooks when it is
created. ``torch.cuda.graph(..., enable_annotations=True)`` enables them for the
length of the capture, tagging the collective kernels through
:func:`torch.cuda.graph_annotations.mark_kernels`; a Cuspy profiling session
enables them while it records, tagging eager kernels through its observer.

Collectives issued inside ``_coalescing_manager`` or a batched ``batch_isend_irecv``
are not annotated: their kernels launch when the group ends, outside any hook.
Collectives on CPU tensors are skipped, since they launch no kernels.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import weakref
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    _set_gated_hooks_enabled,
    HookOpName,
    PostHookArgs,
    PreHookArgs,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    Annotator = Callable[[dict[str, Any]], AbstractContextManager[None]]


logger = logging.getLogger(__name__)

# Registering a hook id replaces its hook, so stay clear of user ids.
_HOOK_ID = 1 << 62

# Names the default NCCL backend passes to record_param_comms.
_COLLECTIVE_NAMES = {
    HookOpName.ALLREDUCE: "all_reduce",
    HookOpName.ALLGATHER: "all_gather",
    HookOpName.ALLTOALL: "all_to_all",
    HookOpName.ALLGATHER_BASE: "all_gather_single",
    HookOpName.REDUCE_SCATTER_BASE: "reduce_scatter_single",
    HookOpName.ALLTOALL_BASE: "all_to_all_single",
}
_IN_PLACE_OPS = {
    HookOpName.BROADCAST,
    HookOpName.ALLREDUCE,
    HookOpName.REDUCE,
    HookOpName.ALLREDUCE_COALESCED,
}


def _format_ranks(ranks: list[int]) -> str:
    # record_param_comms truncates past 30 ranks.
    if len(ranks) > 30:
        head = ", ".join(map(str, ranks[:29]))
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


def _on_cuda(args: PreHookArgs) -> bool:
    tensors = args.input_tensors or args.output_tensors
    return bool(tensors) and tensors[0].is_cuda


def _mark_kernels(metadata: dict[str, Any]) -> AbstractContextManager[None]:
    # The hooks fire on every thread, but only the capturing one may enter the
    # module-global annotation stack.
    if not torch.cuda.is_current_stream_capturing():
        return contextlib.nullcontext()
    from torch.cuda._graph_annotations import mark_kernels

    # Backward attribution would tag whatever autograd node launched the
    # collective, not the collective itself.
    return mark_kernels(metadata, backward=False)


class _GroupHooks:
    def __init__(self, group: dist.ProcessGroup) -> None:
        # The group owns the hooks, so a strong reference would keep it alive.
        self._group = weakref.ref(group)
        self._fields: _GroupFields | None = None
        self._scopes: dict[int, contextlib.ExitStack] = {}

    # A raising hook fails the collective, so annotation errors are only logged.
    def _pre(self, args: PreHookArgs) -> None:
        group = self._group()
        if not _annotators or group is None or not _on_cuda(args):
            return
        scopes = contextlib.ExitStack()
        try:
            if self._fields is None:
                self._fields = _GroupFields(group)
            metadata = collective_metadata(group, args, self._fields)
            # A capture and a profiler can both be open with the same annotator.
            for annotate in dict.fromkeys(_annotators):
                scopes.enter_context(annotate(metadata))
        except Exception:
            logger.exception("Failed to annotate %s", args.name)
            self._exit(scopes)
            return
        self._scopes[args.op_id] = scopes

    def _post(self, args: PostHookArgs) -> None:
        # c10d fires the post hook whenever the pre hook fired, even if the gate
        # closed in between.
        scopes = self._scopes.pop(args.op_id, None)
        if scopes is not None:
            self._exit(scopes)

    @staticmethod
    def _exit(scopes: contextlib.ExitStack) -> None:
        try:
            scopes.close()
        except Exception:
            logger.exception("Failed to close collective annotation")


# One entry per open CollectiveAnnotations.
_annotators: list[Annotator] = []


class CollectiveAnnotations:
    """Tags the collectives of every process group until :meth:`close`.

    ``annotate`` maps a collective's metadata to the scope its kernels launch in;
    it defaults to :func:`~torch.cuda.graph_annotations.mark_kernels` for
    collectives issued from a capturing stream, and a no-op otherwise. Each open
    annotator is entered once per collective, however many instances use it.
    Instances must be closed in the reverse order they were opened.
    """

    def __init__(
        self,
        annotate: Annotator = _mark_kernels,
    ) -> None:
        self._annotate = annotate
        _annotators.append(annotate)
        self._was_enabled = _set_gated_hooks_enabled(True)
        self._closed = False

    def close(self) -> None:
        """Never raises: ``torch.cuda.graph`` calls it before ending the capture."""
        if self._closed:
            return
        self._closed = True
        _set_gated_hooks_enabled(self._was_enabled)
        _annotators.remove(self._annotate)
