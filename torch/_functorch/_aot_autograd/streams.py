import operator
from typing import Any, TYPE_CHECKING, TypeAlias

import torch.fx
import torch.fx.traceback
import torch.utils._pytree as pytree
from torch._dynamo.graph_utils import _get_flat_args
from torch._dynamo.variables.streams import get_current_stream, new_event
from torch.fx.node import map_arg
from torch.utils._runtime_estimation import (
    _FLOAT_TYPES,
    _IGNORE_OPS,
    get_compute_time,
    get_transfer_time,
)


if TYPE_CHECKING:
    from .schemas import ViewAndMutationMeta  # noqa: TC004

from .indexed_dict import IndexedDict


aten = torch.ops.aten

Node: TypeAlias = torch.fx.Node
Graph: TypeAlias = torch.fx.Graph

_SYNC_OPS = (
    torch.ops.streams.record_event.default,
    torch.ops.streams.wait_event.default,
    torch.ops.streams.synchronize_event.default,
    torch.ops.streams.synchronize_device.default,
    torch.ops.streams.synchronize_stream.default,
)


def get_roofline_estimate(node: Node) -> float:
    if node.op != "call_function":
        raise AssertionError(f"non-func node in roofline estimate: {node.op}")

    def map_value(x: Any) -> Any:
        return x.meta.get("value", x) if isinstance(x, Node) else x

    func = node.target
    if func in _IGNORE_OPS:
        return 0.0

    mapped_args = torch.fx.map_arg(node.args, map_value)
    mapped_kwargs = torch.fx.map_arg(node.kwargs, map_value)
    flat_args_kwargs = [map_value(x) for x in _get_flat_args(node, {})]
    flat_outs, _ = pytree.tree_flatten(node.meta.get("value", node))
    out = node.meta.get("value", node)
    out_dtypes = {
        t.dtype
        for t in flat_outs
        if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES
    }

    return (
        max(
            get_transfer_time(flat_args_kwargs, flat_outs),
            get_compute_time(func, mapped_args, mapped_kwargs, out, out_dtypes),
        )
        / 1e6
    )


def is_gradient_acc(node: Node) -> bool:
    return node.meta.get("is_gradient_acc", False)


def is_bwd_node(node: Node) -> bool:
    tag = node.meta.get("partitioner_tag")
    return tag == "is_backward" or tag == "must_be_in_backward"


def get_device(node: Node) -> torch.device:
    return node.meta["val"].device


def get_stream(node: Node) -> int | None:
    maybe_annotation = node.meta.get("custom", None)
    if maybe_annotation is not None:
        return node.meta["custom"].get("stream", None)
    else:
        return None


def get_stream_or_current_stream(node: Node) -> int:
    ind = get_stream(node)
    if ind is None:
        ind = get_current_stream(get_device(node))
    return ind


def set_stream(node: Node, ind: int) -> None:
    if "custom" in node.meta:
        node.meta["custom"].update({"stream": ind})
    else:
        node.meta["custom"] = {"stream": ind}


def insert_record_event_after_node(graph: Graph, node: Node, event_ind: int) -> Node:
    with graph.inserting_after(node):
        node = graph.call_function(
            torch.ops.streams.record_event.default,
            (
                event_ind,
                get_stream_or_current_stream(node),
            ),
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"

    return node


def insert_wait_event_before_node(graph: Graph, node: Node, event_ind: int) -> Node:
    with graph.inserting_before(node):
        node = graph.call_function(
            torch.ops.streams.wait_event.default,
            (
                event_ind,
                get_stream_or_current_stream(node),
            ),
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"

    return node


def populate_stream_timeline(
    stream_to_timeline: dict[int | None, IndexedDict[Node, float]],
    graph: Graph,
    stream_index: int | None,
) -> IndexedDict[Node, float]:
    if stream_index not in stream_to_timeline:
        stream_to_timeline[stream_index] = IndexedDict()
        total_time = 0.0
        for node in graph.nodes:
            # mlazos: not sure if we should include forward here too but don't think it matters
            if (
                node.op == "call_function"
                and is_bwd_node(node)
                and get_stream(node) == stream_index
            ):
                total_time += get_roofline_estimate(node)
                stream_to_timeline[stream_index][node] = (
                    total_time  # NB: total time includes the node's runtime
                )

    return stream_to_timeline[stream_index]


# NB: we start all estimates at 0, estimating the total runtime of each stream with timestamps at each node
# we then try and use these timestamps to estimate when to deallocate tensors used in side streams
# See https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
# for details on the problem being addressed. Rather than using the automatic memory management approach of record_stream
# we attempt to find the point which to deallocate based on the estimated timestamps.
def handle_synced_deallocation(
    graph: Graph,
    stream_to_exec_trace: dict[int | None, IndexedDict[Node, float]],
    node: Node,
    last_usage: Node,
) -> None:
    if not is_bwd_node(node):
        raise AssertionError(
            "synced allocations should only be handled on backward nodes"
        )
    if not is_bwd_node(last_usage):
        raise AssertionError(
            "synced allocations should only be handled on backward nodes"
        )
    allocating_stream = get_stream(node)
    side_stream = get_stream(last_usage)
    if allocating_stream == side_stream:
        raise AssertionError(
            "allocating and side stream should be different for synced deallocations"
        )
    if not torch.cuda.is_available():
        # fallback to record_stream in this case
        with graph.inserting_after(node):
            graph.call_function(
                torch.ops.streams.record_stream.default,
                (
                    node,
                    get_stream_or_current_stream(last_usage),
                ),
                {},
            )
        node.meta["partitioner_tag"] = "must_be_in_backward"

    allocating_stream_trace = populate_stream_timeline(
        stream_to_exec_trace, graph, allocating_stream
    )
    side_stream_trace = populate_stream_timeline(
        stream_to_exec_trace, graph, side_stream
    )

    alloc_ptr = node
    target_side_stream_time = side_stream_trace[last_usage]
    # linear search from first usage of tensor to a point in time after the side stream has finished
    while alloc_ptr is not None:
        alloc_time = allocating_stream_trace[alloc_ptr]

        if alloc_time >= target_side_stream_time:
            break
        elif alloc_time < target_side_stream_time:
            next_ptr = allocating_stream_trace.next_key(alloc_ptr)
            if next_ptr is not None:
                alloc_ptr = next_ptr
            else:
                break

    wait_event = new_event()
    record_node = insert_record_event_after_node(graph, last_usage, wait_event)
    with graph.inserting_after(max(alloc_ptr, record_node)):
        graph.call_function(
            torch.ops.streams.sync_dealloc.default,
            (wait_event, get_stream_or_current_stream(alloc_ptr), node),
            {},
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"


def insert_sync(
    graph: Graph,
    consumer: Node,
    producer: Node,
    node_to_wait_event_ind: dict[Node, int],
) -> None:
    if producer not in node_to_wait_event_ind:
        node_to_wait_event_ind[producer] = new_event()

        insert_record_event_after_node(
            graph, producer, node_to_wait_event_ind[producer]
        )
        insert_wait_event_before_node(graph, consumer, node_to_wait_event_ind[producer])


def assign_backward_streams(gm: torch.fx.GraphModule) -> None:
    """Assigns backward streams to gradient accumulation nodes"""

    # NB: iterate in reverse order to more closely match eager
    # the user node stream will be populated first
    for node in reversed(list(gm.graph.nodes)):
        if is_gradient_acc(node):
            # Accumulation stream selection. Follow the rules from top to bottom to determine the accumulation stream:
            # 1. Match first stream assignment of the first user with a stream
            # 2. Match first stream assignment encountered in the args from left to right
            # This differs from eager in some cases:
            # Specifically the eager code uses the autograd node to determine the stream,
            # crucially this does not necessarily correspond to the FX graph node. For example,
            # in the backward for an add node with a constant we will passthrough and during backward tracing,
            # no op will be added to the FX graph, so our stream assignment will differ in this case.
            gradients = _get_flat_args(node, {})
            users = list(node.users.keys())

            # All gradients will be on same device, they will be coerced if they were not with a .to() node
            for neighbor in users + gradients:
                ind = get_stream(neighbor)
                if ind is not None:
                    set_stream(node, ind)
                    break


def insert_backward_syncs(gm: torch.fx.GraphModule) -> None:
    """Inserts stream syncs for backward nodes if consumer and producer are on different streams"""
    node_to_wait_event_ind: dict[Node, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and is_bwd_node(node):
            flat_args = _get_flat_args(node, {})
            cur_node_stream = get_stream(node)

            for arg in flat_args:
                if arg.op == "call_function" and is_bwd_node(arg):
                    arg_stream = get_stream(arg)
                    if arg_stream != cur_node_stream and get_device(arg).type != "cpu":
                        insert_sync(gm.graph, node, arg, node_to_wait_event_ind)


def sync_deallocations(gm: torch.fx.GraphModule) -> None:
    """Handles https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream"""
    # Note: this is only needed if the last usage of a tensor is on a stream other than
    # the stream the tensor was allocated on

    # an estimated timestamp from the beginning of graph execution (assuming 0 CPU overhead)
    # I think this is fine because you should have large tensors if you're using streams
    # although perhaps I could add a constant 10us per op ahead of the first stream op?
    # a trace of all the nodes running in a given stream
    stream_to_exec_trace: dict[int | None, IndexedDict[Node, float]] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and is_bwd_node(node):
            allocating_stream = get_stream(node)
            users = list(node.users.keys())
            if not users:
                continue
            last_user = max(user for user in users)
            if last_user.op == "output":
                continue
            side_stream = get_stream(last_user)
            if allocating_stream != side_stream:
                handle_synced_deallocation(
                    gm.graph, stream_to_exec_trace, node, last_user
                )


def assign_epilogue_copy_streams(gm: torch.fx.GraphModule) -> None:
    for epi_copy in gm.graph.find_nodes(op="call_function", target=aten.copy_.default):
        arg_stream = get_stream(epi_copy.args[1])
        copy_stream = get_stream(epi_copy)
        if arg_stream != copy_stream:
            set_stream(epi_copy, get_stream_or_current_stream(epi_copy.args[1]))


def populate_fw_metadata_with_stream_indices(
    gm: torch.fx.GraphModule, fw_metadata: "ViewAndMutationMeta"
) -> None:
    """
    Populates fw_metadata.mutated_inp_stream_indices with stream indices from the compiled graph.

    The forward graph outputs are structured as:
    (*mutated_inputs, *user_outputs, *intermediate_bases, *saved_tensors, *saved_symints)

    We extract the stream index for each mutated input from the graph's output node.
    """

    num_mutated_inps = fw_metadata.num_mutated_inp_runtime_indices
    if num_mutated_inps == 0:
        fw_metadata.mutated_inp_stream_indices = []
        return

    # Find the output node in the graph
    output_node = None
    for node in gm.graph.find_nodes(op="output"):
        output_node = node
        break

    if output_node is None:
        raise AssertionError(
            "No output node found in the graph when extracting stream indices"
        )

    # The output node's args[0] is a tuple/list of all outputs
    output_args = output_node.args[0]

    # Extract stream indices for the first num_mutated_inps outputs
    stream_indices = []
    for i in range(num_mutated_inps):
        if i < len(output_args):
            output_arg = output_args[i]
            # Get the stream index from the node metadata
            stream_idx = (
                get_stream(output_arg)
                if isinstance(output_arg, torch.fx.Node)
                else None
            )
            stream_indices.append(stream_idx)
        else:
            stream_indices.append(None)

    fw_metadata.mutated_inp_stream_indices = stream_indices


def _wrap_sync_node(
    gm: torch.fx.GraphModule,
    sync_node: Node,
    deps_before_sync: list[Node],
    visited: set[Node],
) -> tuple[Node, list[Node]]:
    """
    Core logic: wrap a single sync node in control_deps.

    Returns (control_deps_node, passthrough_getitems) where passthrough_getitems
    are the getitem nodes that thread dependencies through the control_deps node.
    ``visited`` is the set of nodes at or before the sync node in graph order,
    used to distinguish pre-sync vs post-sync users.
    """
    from torch._inductor.fx_passes.control_dependencies import (
        _create_subgraph_for_node,
        control_deps,
        get_subgraph_name,
    )

    graph = gm.graph

    # Use dep.users to find deps with uses after the sync — avoids a forward walk.
    deps_with_uses_after_sync = [
        dep
        for dep in deps_before_sync
        if any(user not in visited for user in dep.users)
    ]

    # Create subgraph that executes sync and passes through only used dependencies
    subgraph_module = _create_subgraph_for_node(
        graph, sync_node, deps_with_uses_after_sync
    )
    subgraph_attr_name = get_subgraph_name(gm, sync_node.name)
    setattr(gm, subgraph_attr_name, subgraph_module)

    # Create control_deps call
    # Note: sync nodes (record_event/wait_event) only take int args, no Node args.
    with graph.inserting_before(sync_node):
        get_subgraph = graph.get_attr(subgraph_attr_name)
        control_deps_node = graph.call_function(
            control_deps,
            args=(
                tuple(deps_before_sync),  # additional_deps (all deps for ordering)
                get_subgraph,  # subgraph
                *deps_with_uses_after_sync,  # only pass through deps that are used
            ),
            kwargs={},
        )

    # Mark newly created nodes as visited so subsequent syncs don't
    # misclassify them as "after the sync" during replacement.
    visited.add(get_subgraph)
    visited.add(control_deps_node)

    # The output is (sync_result, *deps_with_uses_after_sync)
    # Create getitem nodes only for dependencies that have uses after sync
    replacements: dict[Node, Node] = {}
    with graph.inserting_after(control_deps_node):
        for i, dep in enumerate(deps_with_uses_after_sync):
            getitem_node = graph.call_function(
                operator.getitem,
                args=(control_deps_node, i + 1),  # +1 because index 0 is sync result
            )
            getitem_node.meta.update(dep.meta)
            replacements[dep] = getitem_node
            visited.add(getitem_node)

    # Replace uses of dependencies that come after sync_node.
    # Use map_arg to handle nested structures (e.g. output node's list args).
    for dep, getitem_node in replacements.items():
        for user in list(dep.users.keys()):
            if user is control_deps_node:
                continue
            if user in visited:
                continue
            # Don't replace forward outputs in the output node — they belong
            # to the forward partition and must not reference backward nodes.
            if user.op == "output" and not is_bwd_node(dep):
                continue

            def _replace(n: Node) -> Node:
                return getitem_node if n is dep else n

            user.args = map_arg(user.args, _replace)
            user.kwargs = map_arg(user.kwargs, _replace)

    # Remove original sync node
    sync_node.replace_all_uses_with(control_deps_node)
    graph.erase_node(sync_node)
    return control_deps_node, list(replacements.values())


def wrap_all_sync_nodes_with_control_deps(gm: torch.fx.GraphModule) -> None:
    """
    Single-pass wrap of all sync nodes in control_deps.

    Iterates through the graph once, accumulating per-stream node lists.
    When a sync node is encountered, it is wrapped using the accumulated deps
    for that stream, then the deps are reset to the control_deps node
    (maintaining the ordering chain for subsequent syncs on the same stream).
    """
    graph = gm.graph
    if len(graph.nodes) == 0:
        raise RuntimeError("Expected a non-empty graph")
    stream_to_nodes: dict[int | None, list[Node]] = {}
    # Maps event_index -> control_deps node that wrapped its record_event,
    # so the corresponding wait_event/synchronize_event can depend on the record.
    event_to_ctrl: dict[int, Node] = {}
    # Maps event_index -> getitem nodes threaded through record_event's control_deps,
    # so synchronize_event can thread them through to subsequent ops.
    event_to_passthrough: dict[int, list[Node]] = {}
    # Maps event_index -> stream that the event was recorded on,
    # so synchronize_event can infer its stream.
    event_to_stream: dict[int, int | None] = {}
    visited: set[Node] = set()
    found_sync = False

    # Walk the node linked-list manually so we can mutate the graph
    # (wrapping sync nodes inserts/erases nodes) without losing our place.
    node = next(iter(graph.nodes))
    while node.op != "root":
        next_node = node.next
        visited.add(node)

        if node.op == "call_function":
            if node.target in _SYNC_OPS:
                # synchronize_device and synchronize_stream block the CPU,
                # so all subsequent kernel launches are host-ordered after
                # them. Treat both as full barriers across all streams.
                if node.target in (
                    torch.ops.streams.synchronize_device.default,
                    torch.ops.streams.synchronize_stream.default,
                ):
                    all_stream_deps: list[Node] = [
                        n for nodes in stream_to_nodes.values() for n in nodes
                    ]
                    if all_stream_deps:
                        found_sync = True
                        _wrap_sync_node(gm, node, all_stream_deps, visited)
                    stream_to_nodes.clear()
                    node = next_node
                    continue

                event_index: int = node.args[0]  # type: ignore[assignment]

                # synchronize_event blocks the CPU thread, so it acts
                # as a barrier across all streams. Collect deps from every
                # stream and reset them all afterward. If the event was
                # recorded externally, thread the graph inputs through so
                # that any post-sync uses depend on the synchronize.
                if node.target is torch.ops.streams.synchronize_event.default:
                    sync_stream: int | None = event_to_stream.get(event_index)
                    all_stream_deps: list[Node] = [
                        n for nodes in stream_to_nodes.values() for n in nodes
                    ]
                    if event_index not in event_to_stream:
                        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
                        deps_before_sync = [*placeholders, *all_stream_deps]
                    else:
                        deps_before_sync = all_stream_deps
                else:
                    sync_stream = node.args[1]  # type: ignore[assignment]
                    deps_before_sync = list(stream_to_nodes.get(sync_stream, ()))
                    # Nodes without explicit stream annotation (custom.stream=None)
                    # run on the current/default stream. Include them when the sync
                    # op references a stream, since the unannotated nodes are
                    # implicitly on that stream.
                    if None in stream_to_nodes and sync_stream is not None:
                        deps_before_sync.extend(stream_to_nodes[None])

                # For wait_event and synchronize_event, add a cross-event
                # dependency on the matching record_event's control_deps node
                # so they cannot be reordered before the record.
                if (
                    node.target
                    in (
                        torch.ops.streams.wait_event.default,
                        torch.ops.streams.synchronize_event.default,
                    )
                    and event_index in event_to_ctrl
                ):
                    deps_before_sync = [
                        event_to_ctrl[event_index],
                        *deps_before_sync,
                    ]

                # For synchronize_event, also include the getitem nodes
                # threaded through record_event's control_deps. This ensures
                # subsequent ops that depend on recorded values get rewired
                # through synchronize_event.
                if (
                    node.target is torch.ops.streams.synchronize_event.default
                    and event_index in event_to_passthrough
                ):
                    deps_before_sync = [
                        *deps_before_sync,
                        *event_to_passthrough[event_index],
                    ]

                if deps_before_sync:
                    found_sync = True
                    ctrl_node, passthrough = _wrap_sync_node(
                        gm, node, deps_before_sync, visited
                    )
                else:
                    ctrl_node = None
                    passthrough: list[torch.fx.Node] = []

                if node.target is torch.ops.streams.record_event.default:
                    event_to_stream[event_index] = sync_stream
                    if ctrl_node is not None:
                        event_to_ctrl[event_index] = ctrl_node
                    event_to_passthrough[event_index] = passthrough

                # Reset: ops between this sync and the next will accumulate
                # fresh. Ordering with prior ops is already enforced because
                # their uses were rewired through getitems from control_deps.
                if node.target is torch.ops.streams.synchronize_event.default:
                    stream_to_nodes.clear()
                else:
                    stream_to_nodes[sync_stream] = []
                    if None in stream_to_nodes:
                        stream_to_nodes[None] = []
            elif "val" in node.meta:
                stream = get_stream(node)
                stream_to_nodes.setdefault(stream, []).append(node)

        node = next_node

    if found_sync:
        gm.recompile()
