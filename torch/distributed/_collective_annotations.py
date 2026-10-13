"""Tag the kernels of c10d collectives with their ``record_param_comms`` fields.

Kineto copies ``record_param_comms`` onto eager NCCL kernels. Replayed CUDA graphs
have no CPU op to copy from, and Cuspy does not see ``record_param_comms``, so
:func:`register_collective_annotation_hooks` provides the same fields from c10d's
per-collective pre/post hooks instead.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from torch._C._distributed_c10d import PreHookArgs, ProcessGroup
from torch.distributed.distributed_c10d import (
    _world,
    Backend,
    BackendConfig,
    get_backend,
    get_process_group_ranks,
)


if TYPE_CHECKING:
    from contextlib import AbstractContextManager


# record_param_comms field name -> value, e.g. "Collective name" -> "allreduce".
CollectiveMetadata: TypeAlias = dict[str, str | int | bool]


class _GroupInfo:
    """A group's fixed ``record_param_comms`` fields, computed once at registration
    rather than on every collective."""

    def __init__(self, group: ProcessGroup) -> None:
        self.group = group
        self.ranks = get_process_group_ranks(group)
        self.rank = group.rank()
        self.fields: CollectiveMetadata = {
            "Group size": len(self.ranks),
            "Process Group Name": group.group_name,
        }
        if group.group_desc:
            self.fields["Process Group Description"] = group.group_desc
        # format_list in util.cpp truncates past 30 entries.
        if len(self.ranks) > 30:
            head = ", ".join(map(str, self.ranks[:29]))
            self.fields["Process Group Ranks"] = f"[{head}, ..., {self.ranks[-1]}]"
        else:
            self.fields["Process Group Ranks"] = f"[{', '.join(map(str, self.ranks))}]"


def _collective_metadata(info: _GroupInfo, args: PreHookArgs) -> CollectiveMetadata:
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
    metadata: CollectiveMetadata = {
        "Collective name": name,
        "In msg nelems": sum(t.numel() for t in inputs),
        "Out msg nelems": sum(t.numel() for t in outputs),
        **info.fields,
        "Is asynchronized op": args.async_op,
    }
    if inputs or outputs:
        # c10::toString(ScalarType), as util.cpp writes it ("Float", "BFloat16"):
        # Tensor.type() is "torch[.cuda].<that name>Tensor".
        tensor_type = (inputs or outputs)[0].type()
        metadata["dtype"] = tensor_type.rsplit(".", 1)[-1].removesuffix("Tensor")
    if op in ("SEND", "RECV"):
        # P2P ops advance a separate counter the group does not expose, so no Seq.
        # Like record_param_comms, "Rank" is the peer's group rank here.
        metadata["Rank"] = args.root
        key = "Dst Rank" if op == "SEND" else "Src Rank"
        metadata[key] = info.ranks[args.root]
    else:
        metadata["Rank"] = info.rank
        # The hook fires before the backend bumps its counter.
        metadata["Seq"] = (
            info.group._get_sequence_number_for_group()  # pyrefly: ignore[missing-attribute]
            + 1
        )
    return metadata


def _runs_cuda_on_nccl(group: ProcessGroup) -> bool:
    """Whether the group is configured to run CUDA collectives on NCCL, whose
    kernels these hooks annotate."""
    device_backends = BackendConfig(get_backend(group)).get_device_backend_map()
    return device_backends.get("cuda") == Backend.NCCL


class CollectiveAnnotationHandle:
    """Hooks set by :func:`register_collective_annotation_hooks`; :meth:`remove`
    unregisters them. Keep the handle until then: its id() is the hook id."""

    def __init__(self, groups: list[ProcessGroup]) -> None:
        self._groups = groups

    def remove(self) -> None:
        """Unregister the hooks from every group they were set on. Idempotent."""
        for group in self._groups:
            group.unregister_pre_hook(id(self))
            group.unregister_post_hook(id(self))
        self._groups = []


def register_collective_annotation_hooks(
    push: Callable[[CollectiveMetadata], None],
    pop: Callable[[], None],
) -> CollectiveAnnotationHandle:
    """Hook every existing process group that runs CUDA collectives on NCCL, so that
    each of its collectives calls ``push(metadata)`` just before it launches its
    kernels and ``pop()`` just after. ``push`` must attach ``metadata`` to every
    kernel the calling thread launches until the matching ``pop``.

    Groups created after this call are not hooked, and collectives inside
    ``_coalescing_manager`` or a batched ``batch_isend_irecv`` are not annotated:
    their kernels launch when the group ends, outside any hook. c10d skips the post
    hook, and so the ``pop``, of a collective that raises.

    Each call gets its own hooks, so registrations can overlap (e.g. a graph capture
    inside a Cuspy window). c10d does not lock its hook maps, so register and remove
    while no other thread is issuing collectives on a hooked group, e.g. between
    steps. Other groups, such as async checkpointing's Gloo group, are never hooked.
    """
    handle = CollectiveAnnotationHandle(
        [group for group in _world.pg_names if _runs_cuda_on_nccl(group)]
    )
    for group in handle._groups:
        info = _GroupInfo(group)

        # The default binds this group's info; a closure would see the last group's.
        def pre_hook(args: PreHookArgs, info: _GroupInfo = info) -> None:
            push(_collective_metadata(info, args))

        group.register_pre_hook(id(handle), pre_hook)
        group.register_post_hook(id(handle), lambda _: pop())
    return handle


def register_graph_capture_annotation_hooks() -> CollectiveAnnotationHandle:
    """Register hooks that tag the kernels a CUDA graph capture records for each
    collective, via ``mark_kernels``. Used by ``torch.cuda.graph``."""
    from torch.cuda._graph_annotations import mark_kernels

    # The scope a collective's push opened, until its pop closes it.
    scopes: list[AbstractContextManager[None]] = []

    def push(metadata: CollectiveMetadata) -> None:
        # Backward attribution would tag whatever autograd node launched the
        # collective, not the collective itself.
        scope = mark_kernels(metadata, backward=False)
        scope.__enter__()
        scopes.append(scope)

    def pop() -> None:
        scopes.pop().__exit__(None, None, None)

    return register_collective_annotation_hooks(push, pop)
