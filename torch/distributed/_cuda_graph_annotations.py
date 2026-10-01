"""Annotate c10d collectives captured into a CUDA graph with their process group.

In eager mode Kineto copies the ``record_param_comms`` metadata of a collective
onto the kernels it launched. A replayed graph has no CPU op to copy from, so its
NCCL kernels carry no process group. ``torch.cuda.graph(..., enable_annotations=True)``
installs these hooks for the length of the capture, so the collective kernels are
tagged through :func:`torch.cuda.graph_annotations.mark_kernels` with the same fields.
"""

from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING

import torch.distributed as dist
from torch._C._distributed_c10d import HookOpName, PostHookArgs, PreHookArgs
from torch.cuda._graph_annotations import mark_kernels


if TYPE_CHECKING:
    from contextlib import AbstractContextManager


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


def collective_metadata(group: dist.ProcessGroup, args: PreHookArgs) -> dict[str, Any]:
    """The ``record_param_comms`` fields Kineto attaches to a collective's kernels."""
    name = _COLLECTIVE_NAMES.get(args.name, args.name.name.lower())
    inputs = args.input_tensors
    outputs = inputs if args.name in _IN_PLACE_OPS else args.output_tensors
    ranks = dist.get_process_group_ranks(group)
    metadata: dict[str, Any] = {
        "Collective name": name,
        "In msg nelems": sum(t.numel() for t in inputs),
        "Out msg nelems": sum(t.numel() for t in outputs),
        "Group size": len(ranks),
        "Process Group Name": group.group_name,
        "Process Group Description": group.group_desc,
        "Process Group Ranks": ranks,
    }
    if inputs or outputs:
        metadata["dtype"] = str((inputs or outputs)[0].dtype).removeprefix("torch.")
    if args.name in (HookOpName.SEND, HookOpName.RECV):
        # P2P ops advance a separate counter the group does not expose, so no Seq.
        if 0 <= args.root < len(ranks):
            key = "Dst Rank" if args.name == HookOpName.SEND else "Src Rank"
            metadata[key] = ranks[args.root]
    else:
        try:
            # The hook fires before the backend bumps its counter.
            metadata["Seq"] = (
                group._get_sequence_number_for_group()  # pyrefly: ignore[missing-attribute]
                + 1
            )
        except RuntimeError:
            pass
    return metadata


class _GroupHooks:
    def __init__(self, group: dist.ProcessGroup) -> None:
        self._group = group
        offset = next(_hook_offsets)
        self._pre_id = _HOOK_ID_BASE + offset
        self._post_id = -_HOOK_ID_BASE - offset
        self._scopes: dict[int, AbstractContextManager[None]] = {}
        group.register_pre_hook(self._pre_id, self._pre)
        try:
            group.register_post_hook(self._post_id, self._post)
        except BaseException:
            group.unregister_pre_hook(self._pre_id)
            raise

    def _pre(self, args: PreHookArgs) -> None:
        # Backward attribution would tag whatever autograd node launched the
        # collective, not the collective itself.
        scope = mark_kernels(collective_metadata(self._group, args), backward=False)
        scope.__enter__()
        self._scopes[args.op_id] = scope

    def _post(self, args: PostHookArgs) -> None:
        scope = self._scopes.pop(args.op_id, None)
        if scope is not None:
            scope.__exit__(None, None, None)

    def close(self) -> None:
        for scope in reversed(self._scopes.values()):
            scope.__exit__(None, None, None)
        self._scopes.clear()
        self._group.unregister_post_hook(self._post_id)
        self._group.unregister_pre_hook(self._pre_id)


class CollectiveAnnotations:
    """Tags collectives on every process group that exists when it is created,
    until :meth:`close`. Groups created afterwards are not hooked."""

    def __init__(self) -> None:
        self._hooks: list[_GroupHooks] = []
        from torch.distributed.distributed_c10d import _world

        try:
            for group in list(_world.pg_map):
                self._hooks.append(_GroupHooks(group))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        while self._hooks:
            self._hooks.pop().close()
