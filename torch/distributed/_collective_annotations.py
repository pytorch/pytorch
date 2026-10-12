"""Tag the kernels of c10d collectives with their ``record_param_comms`` fields.

Kineto copies ``record_param_comms`` onto eager NCCL kernels. Replayed CUDA graphs
have no CPU op to copy from, and Cuspy does not see ``record_param_comms``, so
:func:`register_collective_annotation_hooks` provides the same fields from c10d's
per-collective pre/post hooks instead.
"""

from collections.abc import Callable
from typing import TypeAlias

from torch._C._distributed_c10d import PreHookArgs, ProcessGroup
from torch.distributed.distributed_c10d import _world, get_process_group_ranks


# record_param_comms field name -> value, e.g. "Collective name" -> "allreduce".
CollectiveMetadata: TypeAlias = dict[str, str | int | bool]


def _collective_metadata(group: ProcessGroup, args: PreHookArgs) -> CollectiveMetadata:
    """Mirrors the ``record_param_comms`` fields that ``saveNcclMeta`` in
    torch/csrc/profiler/util.cpp writes for a collective, in the same format. Split
    sizes, Comms Id, the p2p Seq and the global rank start/stride aren't available
    from the hook."""
    op = args.name.name
    # The collective name ProcessGroupNCCL passes to record_param_comms, where it
    # isn't the lowercased op.
    name = {
        "ALLGATHER": "all_gather",
        "ALLTOALL": "all_to_all",
        "ALLGATHER_BASE": "_allgather_base",
        "REDUCE_SCATTER_BASE": "_reduce_scatter_base",
        "ALLTOALL_BASE": "all_to_allv",
    }.get(op, op.lower())
    inputs = args.input_tensors
    in_place = op in ("BROADCAST", "ALLREDUCE", "REDUCE", "ALLREDUCE_COALESCED")
    outputs = inputs if in_place else args.output_tensors
    ranks = get_process_group_ranks(group)
    metadata: CollectiveMetadata = {
        "Collective name": name,
        "In msg nelems": sum(t.numel() for t in inputs),
        "Out msg nelems": sum(t.numel() for t in outputs),
        "Group size": len(ranks),
        "Process Group Name": group.group_name,
    }
    if group.group_desc:
        metadata["Process Group Description"] = group.group_desc
    # format_list in util.cpp truncates past 30 entries.
    if len(ranks) > 30:
        head = ", ".join(map(str, ranks[:29]))
        metadata["Process Group Ranks"] = f"[{head}, ..., {ranks[-1]}]"
    else:
        metadata["Process Group Ranks"] = f"[{', '.join(map(str, ranks))}]"
    metadata["Is asynchronized op"] = args.async_op
    if inputs or outputs:
        metadata["dtype"] = str((inputs or outputs)[0].dtype).removeprefix("torch.")
    if op in ("SEND", "RECV"):
        # P2P ops advance a separate counter the group does not expose, so no Seq.
        # Like record_param_comms, "Rank" is the peer's group rank here.
        metadata["Rank"] = args.root
        if 0 <= args.root < len(ranks):
            key = "Dst Rank" if op == "SEND" else "Src Rank"
            metadata[key] = ranks[args.root]
    else:
        metadata["Rank"] = group.rank()
        try:
            # The hook fires before the backend bumps its counter.
            metadata["Seq"] = (
                group._get_sequence_number_for_group()  # pyrefly: ignore[missing-attribute]
                + 1
            )
        except RuntimeError:
            pass
    return metadata


def _pre_hook(
    group: ProcessGroup, push: Callable[[CollectiveMetadata], None]
) -> Callable[[PreHookArgs], None]:
    def hook(args: PreHookArgs) -> None:
        push(_collective_metadata(group, args))

    return hook


# Registered handles, kept alive until removed so a live handle's id() is never
# reused as another registration's hook id.
_active_handles: dict[int, "CollectiveAnnotationHandle"] = {}


class CollectiveAnnotationHandle:
    """Hooks set by :func:`register_collective_annotation_hooks`; :meth:`remove`
    unregisters them."""

    def __init__(self, groups: list[ProcessGroup]) -> None:
        self._groups = groups

    def remove(self) -> None:
        """Unregister the hooks from every group they were set on. Idempotent."""
        for group in self._groups:
            group.unregister_pre_hook(id(self))
            group.unregister_post_hook(id(self))
        self._groups = []
        _active_handles.pop(id(self), None)


def register_collective_annotation_hooks(
    push: Callable[[CollectiveMetadata], None],
    pop: Callable[[], None],
) -> CollectiveAnnotationHandle:
    """Hook every existing process group so that each collective calls
    ``push(metadata)`` just before it launches its kernels and ``pop()`` just after.
    ``push`` must attach ``metadata`` to every kernel the calling thread launches
    until the matching ``pop``.

    Groups created after this call are not hooked, and collectives inside
    ``_coalescing_manager`` or a batched ``batch_isend_irecv`` are not annotated:
    their kernels launch when the group ends, outside any hook.

    Each call gets its own hooks, so registrations can overlap (e.g. a graph capture
    inside a Cuspy window). c10d does not lock its hook maps, so register and remove
    while no other thread is issuing collectives, e.g. between steps.
    """
    handle = CollectiveAnnotationHandle(list(_world.pg_names))
    _active_handles[id(handle)] = handle
    # If push raises, c10d skips the collective and its post hook, so pushes and
    # pops always pair up.
    for group in handle._groups:
        group.register_pre_hook(id(handle), _pre_hook(group, push))
        group.register_post_hook(id(handle), lambda _: pop())
    return handle
