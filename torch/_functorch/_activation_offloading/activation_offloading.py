"""
Activation offloading for memory optimization in (more like post) partitioners.

This module provides functionality to offload activations to CPU during forward pass
and reload them during backward pass, reducing GPU memory usage.
"""

import logging
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._dynamo.variables.streams import get_current_stream, new_event, new_stream
from torch._inductor import config as inductor_config
from torch._inductor.fx_passes.overlap_scheduling import benchmark_node, is_compute_node
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..partitioners import _size_of, get_default_op_list, OpTypes


log: logging.Logger = logging.getLogger(__name__)


# Node name prefixes for offload/reload operations
CPU_OFFLOAD_PREFIX = "cpu_offload_"
GPU_RELOAD_PREFIX = "gpu_reload_"


@dataclass
class ReloadNodeInfo:
    """
    Information about backward reload related nodes for each reload operation.

    Pattern: fork → wait_stream → device_put → record_event → join → wait_event

    This pattern is divided into two logical groups for optimization purposes:
    - Reload group (fork → wait_stream → device_put → record_event → join):
      Performs the actual asynchronous data transfer on a separate stream.
      These nodes can be moved earlier in the graph to overlap with computation.
    - Wait group (wait_event):
      Synchronization point that blocks until the data transfer completes.
      This must remain at the point where the reloaded data is first needed.
    """

    reload_group_nodes: list[fx.Node]
    wait_event_node: fx.Node
    transfer_size_bytes: int
    transfer_time_ms: float


@dataclass
class ReloadQueueEntry:
    """
    Entry in the reload queue for prefetch scheduling.

    Attributes:
        pattern: The reload pattern information
        remaining_time_ms: Remaining overlap time needed in milliseconds
    """

    pattern: ReloadNodeInfo
    remaining_time_ms: float


def offload_activation_fw(graph: fx.Graph) -> None:
    """
    Forward pass modification for CPU offloading, by inserting offloaindg nodes.

    Args:
        graph: The forward graph to modify
    """

    effective_users_cache: dict[fx.Node, OrderedSet[fx.Node]] = {}
    op_types: OpTypes = get_default_op_list()

    def find_all_effective_users(node: fx.Node) -> OrderedSet[fx.Node]:
        """
        Find all effective users of a node, where view ops extend the lifetime of the
        original node. If a user is a view op, recursively find  users of the view.
        """
        if node in effective_users_cache:
            return effective_users_cache[node]

        effective_users: OrderedSet[fx.Node] = OrderedSet()
        for user in node.users.keys():
            if user.op == "output":
                continue
            effective_users.add(user)
            if op_types.is_view(user):
                effective_users.update(find_all_effective_users(user))

        effective_users_cache[node] = effective_users

        return effective_users

    output_node: fx.Node = graph.find_nodes(op="output")[0]
    fwd_outputs: tuple[fx.Node] = output_node.args[
        0
    ]  # pyrefly: ignore [bad-assignment]
    node_to_offload: dict[fx.Node, fx.Node] = dict()
    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    for node in fwd_outputs:
        if node.meta.get("saved_for_offloading", False) is False:
            continue

        # Find insertion point, which is the last use
        all_effective_users: OrderedSet[fx.Node] = find_all_effective_users(node)
        if all_effective_users := find_all_effective_users(node):
            last_user = max(all_effective_users, key=lambda n: node_to_index[n])
        else:
            last_user: fx.Node = node

        # Insert the CPU offload operation after the last user
        with graph.inserting_after(last_user):
            cpu_node: fx.Node = graph.call_function(
                torch.ops.prims.device_put.default,
                args=(node, torch.device("cpu")),
                kwargs={"non_blocking": True},
                name=CPU_OFFLOAD_PREFIX + str(node.name),
            )
            cpu_node.meta["val"] = node.meta["val"].to(torch.device("cpu"))
            cpu_node.meta["tensor_meta"] = extract_tensor_metadata(cpu_node.meta["val"])

        node_to_offload[node] = cpu_node

    # Update the return node args
    output_node.update_arg(
        0, tuple(node_to_offload.get(node, node) for node in fwd_outputs)
    )


def offload_activation_bw(graph: fx.Graph) -> None:
    """
    Backward pass modification for GPU reloading, by inserting reloaindg nodes.

    Args:
        graph: The backward graph to modify
    """

    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }
    output_node: fx.Node = graph.find_nodes(op="output")[0]

    for node in graph.find_nodes(op="placeholder"):
        if node.meta.get("saved_for_offloading", False) is False:
            continue

        # Find insertion point, which is the first use or output node if no users
        # The later should not happen, but inserting before output node is safe
        insert_point: fx.Node = (
            min(node.users.keys(), key=lambda n: node_to_index[n])
            if node.users
            else output_node
        )

        # Insert the GPU reload operation before the first user
        original_device: torch.Device = node.meta["original_device"]
        with graph.inserting_before(insert_point):
            gpu_node: fx.Node = graph.call_function(
                torch.ops.prims.device_put.default,
                args=(node, original_device),
                kwargs={"non_blocking": True},
                name=str(node.name).replace(CPU_OFFLOAD_PREFIX, GPU_RELOAD_PREFIX),
            )
            gpu_node.meta["val"] = node.meta["val"].to(original_device)
            gpu_node.meta["tensor_meta"] = extract_tensor_metadata(gpu_node.meta["val"])

        # Replace all uses of the CPU tensor with the GPU tensor
        for user in list(node.users.keys()):
            if user != gpu_node:
                user.replace_input_with(node, gpu_node)


def can_offload(
    node: fx.Node,
    fwd_outputs: OrderedSet[fx.Node],
    model_outputs: OrderedSet[fx.Node],
    static_lifetime_input_nodes: OrderedSet[fx.Node],
) -> bool:
    """
    Determine if a node can be offloaded to CPU.

    Args:
        node: The node to check
        fwd_outputs: Forward module outputs, including model outputs and activations
        model_outputs: Model outputs
    """
    log.debug(f"Checking node {node.name} for offloading...")  # noqa: G004

    op_types: OpTypes = get_default_op_list()

    if node not in fwd_outputs:
        log.debug("\tSkipped! Can only offload nodes in fwd_module_outputs.")
        return False
    if node in model_outputs:
        log.debug("\tSkipped! Cannot offload model outputs.")
        return False
    if node in static_lifetime_input_nodes:
        log.debug("\tSkipped! Cannot offload static input nodes.")
        return False
    if op_types.is_view(node) or node.target == operator.getitem:
        log.debug("\tSkipped! Cannot offload views of getitems.")
        return False
    if hasattr(node, "meta") and "val" in node.meta:
        if (
            isinstance(val := node.meta["val"], torch.Tensor)
            and not val.is_contiguous()
        ):
            log.debug("\tSkipped! Cannot offload non-contiguous tensors.")
            return False

    log.debug("\tGood!")
    return True


def choose_offload_sets(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: OrderedSet[fx.Node],
) -> bool:
    """
    Decide which nodes will be offloaded based on the marked nodes and feasibility.
    Marks nodes with "saved_for_offloading" if they should and can be offloaded.

    Args:
        fwd_module: Forward graph module
        bwd_module: Backward graph module
        num_fwd_outputs: Number of forward outputs

    Returns:
        bool: Whether activation offloading should be performed
    """

    fwd_outputs: OrderedSet[fx.Node] = OrderedSet(
        fwd_module.graph.find_nodes(op="output")[0].args[0]
    )
    model_outputs: OrderedSet[fx.Node] = OrderedSet(
        fwd_module.graph.find_nodes(op="output")[0].args[0][:num_fwd_outputs]
    )

    should_perform_offloading = False
    for node in fwd_module.graph.nodes:
        if node.meta.get("should_offload", False) is False:
            continue

        if can_offload(node, fwd_outputs, model_outputs, static_lifetime_input_nodes):
            node.meta["saved_for_offloading"] = True
            node.meta["original_device"] = node.meta["val"].device
            should_perform_offloading = True

    return should_perform_offloading


def offload_chosen_sets(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """
    Add offload and reload nodes to the forward and backward graphs.
    This function adds device_put operations without any stream handling.

    Args:
        fwd_module: Forward module graph
        bwd_module: Backward module graph
    """

    # Add offload nodes in forward graph
    offload_activation_fw(fwd_module.graph)

    # Update backward graph inputs to be offloaded tensors
    bwd_inputs: dict[str, fx.Node] = {
        node.name: node for node in bwd_module.graph.find_nodes(op="placeholder")
    }
    for fwd_node in fwd_module.graph.find_nodes(op="output")[0].args[0]:
        if CPU_OFFLOAD_PREFIX not in fwd_node.name:
            continue

        bwd_node: fx.Node = bwd_inputs[fwd_node.name.replace(CPU_OFFLOAD_PREFIX, "")]
        with bwd_module.graph.inserting_after(bwd_node):
            bwd_offload_node: fx.Node = bwd_module.graph.placeholder(name=fwd_node.name)

        bwd_offload_node.meta.update(fwd_node.meta)
        bwd_offload_node.meta["saved_for_offloading"] = True
        bwd_offload_node.meta["original_device"] = bwd_node.meta["val"].device
        bwd_node.replace_all_uses_with(bwd_offload_node)
        bwd_module.graph.erase_node(bwd_node)

    # Add reload nodes in backward graph
    offload_activation_bw(bwd_module.graph)


def get_default_stream_id(nodes: list[fx.Node]) -> int:
    first_node: fx.Node = nodes[0]
    source_node: fx.Node = first_node.args[0]  # type: ignore[assignment]
    device: torch.device = source_node.meta["val"].device
    return get_current_stream(device)


def add_forward_offload_stream_ops(graph: fx.Graph) -> None:
    """
    Add stream operations for forward pass CPU offloading.

    Pattern: record_event → fork → wait_event → device_put → record_event_2 → join → wait_event_2

    This ensures that:
    1. Offloading waits for the last use to complete (record_event on default stream)
    2. Offloading happens on a separate stream (fork → wait_event → device_put)
    3. Execution returns to the default stream after offloading and
       waits for offload to complete (record_event_2 → join → wait_event_2)

    NOTE: For stream optimization and overlapping compute with communication,
          the "wait_event_2" ops can be sinked to the end of the graph.

    Args:
        graph: The forward graph to modify
    """

    # Find all CPU offload nodes
    offload_nodes: list[fx.Node] = [
        node
        for node in graph.nodes
        if CPU_OFFLOAD_PREFIX in node.name and node.op == "call_function"
    ]
    if not offload_nodes:
        return

    # Get default stream id and offload stream id
    default_stream_id: int = get_default_stream_id(offload_nodes)
    offload_stream_id: int = new_stream()

    for offload_node in offload_nodes:
        offload_ready_event_id: int = new_event()
        offload_completion_event_id: int = new_event()

        with graph.inserting_before(offload_node):
            # Record event on default stream to ensure last use completes
            graph.call_function(
                torch.ops.streams.record_event.default,
                args=(offload_ready_event_id, default_stream_id),
            )
            # Fork to offload stream
            graph.call_function(
                torch.ops.streams.fork.default,
                args=(default_stream_id, offload_stream_id),
                name=f"stream_in_{offload_node.name}",
            )
            # Wait for the event on offload stream
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(offload_ready_event_id, offload_stream_id),
            )
        with graph.inserting_after(offload_node):
            # Record event on offload stream after device_put completes
            record_event_node = graph.call_function(
                torch.ops.streams.record_event.default,
                args=(offload_completion_event_id, offload_stream_id),
            )
        with graph.inserting_after(record_event_node):
            # Join back to default stream
            join_node = graph.call_function(
                torch.ops.streams.join.default,
                args=(offload_stream_id, default_stream_id),
                name=f"stream_out_{offload_node.name}",
            )
        with graph.inserting_after(join_node):
            # Wait for the offload to complete on default stream
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(offload_completion_event_id, default_stream_id),
            )


def add_backward_reload_stream_ops(graph: fx.Graph) -> None:
    """
    Add stream operations for backward pass GPU reloading.

    Pattern: fork → wait_stream → device_put → record_event → join → wait_event

    This ensures that:
    1. Reloading doesn't start prematurely (fork → wait_stream)
    2. Reloading happens on a separate stream (device_put)
    3. First use waits for reload completion (record_event → join → wait_event)

    NOTE: The pattern consists of two logical groups:
          - First group (fork → wait_stream → device_put → record_event → join):
            Performs asynchronous data transfer on a separate stream
          - Second group (wait_event):
            Data transfer completion check when the data is actually needed

          For prefetch optimization, the first group can be moved earlier in the graph
          to overlap computation with data transfer, while the wait_event must remain
          at its current position to prevent blocking computation unnecessarily.

    Args:
        graph: The backward graph to modify
    """

    # Find all GPU reload nodes
    reload_nodes: list[fx.Node] = [
        node
        for node in graph.nodes
        if GPU_RELOAD_PREFIX in node.name and node.op == "call_function"
    ]
    if not reload_nodes:
        return

    # Get default stream id and offload stream id
    default_stream_id: int = get_default_stream_id(reload_nodes)
    reload_stream_id: int = new_stream()

    for reload_node in reload_nodes:
        event_id: int = new_event()

        with graph.inserting_before(reload_node):
            # Fork to reload stream
            graph.call_function(
                torch.ops.streams.fork.default,
                args=(default_stream_id, reload_stream_id),
                name=f"stream_in_{reload_node.name}",
            )
            # Wait for default stream to prevent premature reloading
            graph.call_function(
                torch.ops.streams.wait_stream.default,
                args=(reload_stream_id, default_stream_id),
            )
        with graph.inserting_after(reload_node):
            # Record event on reload stream after device_put
            record_event_node = graph.call_function(
                torch.ops.streams.record_event.default,
                args=(event_id, reload_stream_id),
            )
        with graph.inserting_after(record_event_node):
            # Join back to default stream
            join_node = graph.call_function(
                torch.ops.streams.join.default,
                args=(reload_stream_id, default_stream_id),
                name=f"stream_out_{reload_node.name}",
            )
        with graph.inserting_after(join_node):
            # Wait for the event on default stream
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(event_id, default_stream_id),
            )


def put_offload_nodes_on_separate_stream(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """
    Add stream and event related operations around offload nodes.

    Args:
        fwd_module: Forward module graph
        bwd_module: Backward module graph
    """

    add_forward_offload_stream_ops(fwd_module.graph)
    add_backward_reload_stream_ops(bwd_module.graph)


def _validate_pattern_nodes(
    fork_node: fx.Node,
    wait_stream_node: fx.Node,
    device_put_node: fx.Node,
    record_event_node: fx.Node,
    join_node: fx.Node,
    wait_event_node: fx.Node,
) -> None:
    """
    Validate that the pattern nodes match the expected structure.

    Raises ValueError if any node doesn't match expectations.
    """

    if not (
        fork_node.op == "call_function"
        and fork_node.name == f"stream_in_{device_put_node.name}"
        and fork_node.target == torch.ops.streams.fork.default
    ):
        raise ValueError("Expected fork node two nodes before device_put node")

    if not (
        wait_stream_node.op == "call_function"
        and wait_stream_node.target == torch.ops.streams.wait_stream.default
    ):
        raise ValueError("Expected wait_stream node one node before device_put node")

    if not (
        record_event_node.op == "call_function"
        and record_event_node.target == torch.ops.streams.record_event.default
    ):
        raise ValueError("Expected record_event node one node after device_put node")

    if not (
        join_node.op == "call_function"
        and join_node.name == f"stream_out_{device_put_node.name}"
        and join_node.target == torch.ops.streams.join.default
    ):
        raise ValueError("Expected join node two nodes after device_put node")

    if not (
        wait_event_node.op == "call_function"
        and wait_event_node.target == torch.ops.streams.wait_event.default
    ):
        raise ValueError("Expected wait_event node three nodes after device_put node")


def _calculate_transfer_size(device_put_node: fx.Node) -> int:
    """Calculate the size in bytes of data being transferred."""

    return _size_of(device_put_node.args[0])  # pyrefly: ignore [bad-argument-type]


def _estimate_transfer_time_in_ms(transfer_size_bytes: int) -> float:
    """Estimate transfer time in milliseconds based on size and bandwidth."""

    return transfer_size_bytes / (1024**3) * 1_000 / inductor_config.cpu_gpu_bw


def identify_reload_patterns(
    graph: fx.Graph, nodes_list: list[fx.Node], node_to_idx: dict[fx.Node, int]
) -> dict[fx.Node, ReloadNodeInfo]:
    """
    Identify backward reload patterns in the graph.

    Pattern: fork → wait_stream → device_put → record_event → join → wait_event

    Returns a dict mapping device_put node to ReloadNodeInfo containing:
    - reload_group_nodes: fork → wait_stream → device_put → record_event → join
    - wait_event_node: the wait_event node
    - transfer_size_bytes: size of data being transferred
    - transfer_time_ms: estimated transfer time in milliseconds
    """
    patterns: dict[fx.Node, ReloadNodeInfo] = {}

    # Find all GPU reload device_put nodes
    reload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if GPU_RELOAD_PREFIX in node.name
    ]

    # Extract patterns for each reload device_put node
    for reload_node in reload_nodes:
        reload_node_idx: int = node_to_idx[reload_node]

        fork_node: fx.Node = nodes_list[reload_node_idx - 2]
        wait_stream_node: fx.Node = nodes_list[reload_node_idx - 1]
        record_event_node: fx.Node = nodes_list[reload_node_idx + 1]
        join_node: fx.Node = nodes_list[reload_node_idx + 2]
        wait_event_node: fx.Node = nodes_list[reload_node_idx + 3]

        # Validate the nodes are what we expect
        _validate_pattern_nodes(
            fork_node,
            wait_stream_node,
            reload_node,
            record_event_node,
            join_node,
            wait_event_node,
        )

        # Calculate transfer size and time
        transfer_size_bytes: int = _calculate_transfer_size(reload_node)
        transfer_time_ms: float = _estimate_transfer_time_in_ms(transfer_size_bytes)

        patterns[reload_node] = ReloadNodeInfo(
            reload_group_nodes=[
                fork_node,
                wait_stream_node,
                reload_node,
                record_event_node,
                join_node,
            ],
            wait_event_node=wait_event_node,
            transfer_size_bytes=transfer_size_bytes,
            transfer_time_ms=transfer_time_ms,
        )

    return patterns


def reorder_for_prefetch(
    graph: fx.Graph,
    nodes_list: list[fx.Node],
    reload_patterns: dict[fx.Node, ReloadNodeInfo],
) -> None:
    """
    Reorder nodes to prefetch reload operations by directly manipulating the graph.

    This follows the algorithm as follows:
    - Go through nodes in reverse order
    - When encountering a reload pattern, add it to a queue with its transfer time
    - When encountering a compute node, use its runtime to satisfy overlap requirements
    - Place reload patterns when their overlap requirement is satisfied
    - When encountering placeholder nodes, flush queue as reloads cannot move before inputs
    """

    # Build a set of all nodes in reload groups for quick lookup
    reload_group_nodes_set: set[fx.Node] = set()
    for pattern in reload_patterns.values():
        reload_group_nodes_set.update(pattern.reload_group_nodes)

    # Queue to hold reload group nodes waiting to be placed (FIFO)
    reload_queue: list[ReloadQueueEntry] = []

    # Loop through nodes in reverse
    for node in reversed(nodes_list):
        if node.op == "output":
            continue
        elif node.op == "placeholder":
            # Flush queue - place all remaining reloads after the last placeholder
            while reload_queue:
                entry: ReloadQueueEntry = reload_queue.pop(0)
                for reload_group_node in reversed(entry.pattern.reload_group_nodes):
                    node.append(reload_group_node)
            break
        elif node in reload_patterns:
            pattern: ReloadNodeInfo = reload_patterns[node]
            reload_queue.append(
                ReloadQueueEntry(
                    pattern=pattern, remaining_time_ms=pattern.transfer_time_ms
                )
            )
        elif node in reload_group_nodes_set:
            continue
        else:
            if not reload_queue:
                continue
            compute_runtime_ms: float = (
                benchmark_node(node) if is_compute_node(node) else 0
            )
            reload_queue[0].remaining_time_ms -= compute_runtime_ms

            # Pop and place reload if its remaining time is satisfied (<= 0)
            if reload_queue[0].remaining_time_ms <= 0:
                entry: ReloadQueueEntry = reload_queue.pop(0)
                for reload_group_node in entry.pattern.reload_group_nodes:
                    node.prepend(reload_group_node)


def activation_offload_sink_wait(fwd_module: fx.GraphModule) -> None:
    """
    Sink wait_event operations for offload completion to the end of the graph.

    This function identifies wait_event nodes for offload completion and moves them
    to the end of the graph, allowing computation to overlap with offload operations.

    Args:
        fwd_module: Forward module graph
    """
    graph: fx.Graph = fwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)
    node_to_idx: dict[fx.Node, int] = {node: idx for idx, node in enumerate(nodes_list)}

    # Find all CPU offload device_put nodes
    offload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if CPU_OFFLOAD_PREFIX in node.name
    ]

    # Collect all wait_event nodes that need to be moved
    wait_nodes_to_sink: list[fx.Node] = []
    for offload_node in offload_nodes:
        offload_idx: int = node_to_idx[offload_node]
        wait_event_node: fx.Node = nodes_list[offload_idx + 3]

        # Validate it's actually a wait_event node
        if not (
            wait_event_node.op == "call_function"
            and wait_event_node.target == torch.ops.streams.wait_event.default
        ):
            raise ValueError(
                f"Expected wait_event node three positions after {offload_node.name}"
            )

        wait_nodes_to_sink.append(wait_event_node)

    # Find the output node, and move all wait_event nodes to just before the output node
    output_node: fx.Node = graph.find_nodes(op="output")[0]
    for wait_node in wait_nodes_to_sink:
        output_node.prepend(wait_node)


def activation_reload_prefetch(bwd_module: fx.GraphModule) -> None:
    """
    Prefetch backward reload operations by moving them earlier in the graph
    to overlap communication with computation.

    This function identifies backward reload patterns (fork → wait_stream → device_put →
    record_event → join) and moves them earlier in the execution order to overlap
    the data transfer with computation, while keeping the wait_event at its original
    position.

    Args:
        bwd_module: Backward module graph
    """
    graph: fx.Graph = bwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)
    node_to_idx: dict[fx.Node, int] = {node: idx for idx, node in enumerate(nodes_list)}

    # Step 1: Identify reload patterns
    reload_patterns: dict[fx.Node, ReloadNodeInfo] = identify_reload_patterns(
        graph, nodes_list, node_to_idx
    )

    # Step 2: Reorder nodes by directly manipulating the graph
    reorder_for_prefetch(graph, nodes_list, reload_patterns)


def enable_activation_offloading(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: OrderedSet[fx.Node],
) -> None:
    """
    Main entry point for activation offloading.

    Args:
        fwd_module: Forward module graph
        bwd_module: Backward module graph
        num_fwd_outputs: Number of forward outputs
    """

    # Step 1: Decide which nodes to offload and mark them
    should_perform_offloading: bool = choose_offload_sets(
        fwd_module,
        bwd_module,
        num_fwd_outputs,
        static_lifetime_input_nodes,
    )
    if not should_perform_offloading:
        return

    # Step 2: Add offload and reload nodes to the graphs
    offload_chosen_sets(fwd_module, bwd_module)

    # Step 3: Put offload nodes on separate stream if configured
    if config.activation_offload_separate_stream:
        put_offload_nodes_on_separate_stream(fwd_module, bwd_module)
        if config.activation_offload_sink_wait:
            activation_offload_sink_wait(fwd_module)
        if config.activation_reload_prefetch:
            activation_reload_prefetch(bwd_module)

    fwd_module.graph.lint()
    bwd_module.graph.lint()
