"""Tag the kernels of c10d collectives with their ``record_param_comms`` fields.

Kineto copies ``record_param_comms`` onto eager NCCL kernels. Replayed CUDA graphs
have no CPU op to copy from, and Cuspy does not see ``record_param_comms``, so
:func:`register_collective_annotation_hooks` provides the same fields from c10d's
per-collective pre/post hooks instead, until
:func:`unregister_collective_annotation_hooks` removes them.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from torch._C._distributed_c10d import (
    HookOpName,
    PostHookArgs,
    PreHookArgs,
    ProcessGroup,
)
from torch.distributed.distributed_c10d import (
    _world,
    Backend,
    BackendConfig,
    get_backend,
    get_global_rank,
    get_process_group_ranks,
)


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from torch.cuda.graphs import graph
    from torch.profiler._cuspy.observers.base import CuspyObserver


# The fields saveNcclMeta in torch/csrc/profiler/util.cpp writes onto a
# collective's record_param_comms event, e.g. "Collective name" -> "all_reduce".
CollectiveMetadata: TypeAlias = dict[str, str | int | bool]


# The collective name for each op that fires c10d hooks: the collective's family,
# whatever API variant issued it (list, _base, coalesced), so names don't depend
# on the API or the backend.
_COLLECTIVE_NAMES: dict[HookOpName, str] = {
    HookOpName.SEND: "send",
    HookOpName.RECV: "recv",
    HookOpName.BROADCAST: "broadcast",
    HookOpName.ALLREDUCE: "all_reduce",
    HookOpName.ALLREDUCE_COALESCED: "all_reduce",
    HookOpName.REDUCE: "reduce",
    HookOpName.ALLGATHER: "all_gather",
    HookOpName.ALLGATHER_BASE: "all_gather",
    HookOpName.ALLGATHER_INTO_TENSOR_COALESCED: "all_gather",
    HookOpName.REDUCE_SCATTER: "reduce_scatter",
    HookOpName.REDUCE_SCATTER_BASE: "reduce_scatter",
    HookOpName.REDUCE_SCATTER_TENSOR_COALESCED: "reduce_scatter",
    HookOpName.ALLTOALL: "all_to_all",
    HookOpName.ALLTOALL_BASE: "all_to_all",
    HookOpName.GATHER: "gather",
    HookOpName.SCATTER: "scatter",
    HookOpName.BARRIER: "barrier",
}

# Ops whose outputs are their inputs, so the hook gets no output tensors.
_IN_PLACE_OPS = (
    HookOpName.BROADCAST,
    HookOpName.ALLREDUCE,
    HookOpName.ALLREDUCE_COALESCED,
    HookOpName.REDUCE,
)


def _process_group_fields(group: ProcessGroup) -> CollectiveMetadata:
    """The group-level subset of :func:`_nccl_collective_metadata`: the
    ``saveNcclMeta`` fields that are fixed per group (size, name, description and
    ranks). Computed once per group at registration, since formatting the rank list
    costs O(group size) and would otherwise run on every collective."""
    ranks = get_process_group_ranks(group)
    fields: CollectiveMetadata = {
        "Group size": len(ranks),
        "Process Group Name": group.group_name,
    }
    if group.group_desc:
        fields["Process Group Description"] = group.group_desc
    # format_list in util.cpp truncates past 30 entries.
    if len(ranks) > 30:
        head = ", ".join(map(str, ranks[:29]))
        fields["Process Group Ranks"] = f"[{head}, ..., {ranks[-1]}]"
    else:
        fields["Process Group Ranks"] = f"[{', '.join(map(str, ranks))}]"
    return fields


def _nccl_collective_metadata(
    group: ProcessGroup, group_fields: CollectiveMetadata, args: PreHookArgs
) -> CollectiveMetadata:
    """Mirrors the fields ``saveNcclMeta`` writes for a ProcessGroupNCCL
    collective, in the same format except for the collective name (see
    ``_COLLECTIVE_NAMES``), with the group's from ``group_fields``. Split sizes,
    Comms Id, the p2p Seq and the global rank start/stride aren't available from
    the hook."""
    op = args.name
    inputs = args.input_tensors
    outputs = inputs if op in _IN_PLACE_OPS else args.output_tensors
    metadata: CollectiveMetadata = {
        # An op c10d adds later gets its lowercased name rather than raising,
        # which would fail the collective.
        "Collective name": _COLLECTIVE_NAMES.get(op, op.name.lower()),
        "In msg nelems": sum(t.numel() for t in inputs),
        "Out msg nelems": sum(t.numel() for t in outputs),
        **group_fields,
        "Is asynchronized op": args.async_op,
    }
    # c10::toString(ScalarType), as util.cpp writes it ("Float", "BFloat16").
    # Tensor.type() is built as "<backend>." + that name + "Tensor"
    # (options_to_string in torch/csrc/utils/tensor_types.cpp). Like
    # ProcessGroupNCCL, use the first input's dtype, and Byte with no tensors.
    tensors = inputs or outputs
    metadata["dtype"] = (
        tensors[0].type().rsplit(".", 1)[-1].removesuffix("Tensor")
        if tensors
        else "Byte"
    )
    if op in (HookOpName.SEND, HookOpName.RECV):
        # P2P ops advance a separate counter the group does not expose, so no Seq.
        # Like record_param_comms, "Rank" is the peer's group rank here.
        metadata["Rank"] = args.root
        key = "Dst Rank" if op == HookOpName.SEND else "Src Rank"
        metadata[key] = get_global_rank(group, args.root)
    else:
        metadata["Rank"] = group.rank()
        # The hook fires before the backend bumps its counter.
        metadata["Seq"] = (
            group._get_sequence_number_for_group()  # pyrefly: ignore[missing-attribute]
            + 1
        )
    return metadata


def _pgs_to_annotate() -> list[ProcessGroup]:
    """Existing process groups configured to run CUDA collectives on NCCL, whose
    kernels the hooks annotate."""
    return [
        group
        for group in _world.pg_names
        if BackendConfig(get_backend(group)).get_device_backend_map().get("cuda")
        == Backend.NCCL
    ]


def register_collective_annotation_hooks(
    annotator: "graph | CuspyObserver",
    push: Callable[[CollectiveMetadata], None],
    pop: Callable[[], object],
) -> None:
    """Hook every existing process group that runs CUDA collectives on NCCL, so that
    each of its collectives calls ``push(metadata)`` just before it launches its
    kernels and ``pop()`` just after. ``push`` must attach ``metadata`` to every
    kernel the calling thread launches until the matching ``pop``.

    ``annotator`` is the object annotating the kernels, a CUDA graph capture or a
    Cuspy observer. The hooks are keyed on it (the hook id is ``id(annotator)``), so
    registrations by different annotators can overlap (e.g. a graph capture inside
    a Cuspy window). ``annotator`` must stay alive until
    :func:`unregister_collective_annotation_hooks` is called with it.

    Groups created after this call are not hooked, and collectives inside
    ``_coalescing_manager`` or a batched ``batch_isend_irecv`` are not annotated:
    their kernels launch when the group ends, outside any hook. c10d skips the post
    hook, and so the ``pop``, of a collective that raises.

    c10d does not lock its hook maps, so register and unregister while no other
    thread is issuing collectives on a hooked group, e.g. between steps. Other
    groups, such as async checkpointing's Gloo group, are never touched.
    """

    def post_hook(_: PostHookArgs) -> None:
        pop()

    for group in _pgs_to_annotate():
        group_fields = _process_group_fields(group)

        # The defaults bind this group; a closure would see the last group.
        def pre_hook(
            args: PreHookArgs,
            group: ProcessGroup = group,
            group_fields: CollectiveMetadata = group_fields,
        ) -> None:
            push(_nccl_collective_metadata(group, group_fields, args))

        group.register_pre_hook(id(annotator), pre_hook)
        group.register_post_hook(id(annotator), post_hook)


def unregister_collective_annotation_hooks(
    annotator: "graph | CuspyObserver",
) -> None:
    """Remove the hooks :func:`register_collective_annotation_hooks` set for
    ``annotator``. A no-op on groups ``annotator`` never hooked, so calling it twice,
    or without registering, is safe."""
    for group in _pgs_to_annotate():
        group.unregister_pre_hook(id(annotator))
        group.unregister_post_hook(id(annotator))


def register_graph_capture_annotation_hooks(capture: "graph") -> None:
    """Register hooks, keyed on ``capture``, that tag the kernels the capture
    records for each collective, via ``mark_kernels``. Used by
    ``torch.cuda.graph``."""
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

    register_collective_annotation_hooks(capture, push, pop)
