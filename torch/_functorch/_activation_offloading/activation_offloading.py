"""Activation offloading for memory optimization during compilation.

This module provides functionality to offload activations to CPU during the forward
pass and reload them during the backward pass, reducing GPU memory usage. It can be
applied to graphs produced by both AOT Autograd partitioners and make_fx-based tracing.
"""

import logging
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._functorch._activation_offloading.offload_ops import (  # noqa: F401 -- registers ao::offload, ao::reload, ao::wait_tensor ops
    offload,
    reload,
    wait_tensor,
)
from torch._inductor.fx_passes.overlap_scheduling import benchmark_node, is_compute_node
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..partitioners import _size_of, get_default_op_list, OpTypes


log: logging.Logger = logging.getLogger(__name__)


# Node name prefixes for offload/reload operations
# NOTE: right now we are using these prefixes as identifiers for offload/reload
CPU_OFFLOAD_PREFIX = "cpu_offload_"
GPU_RELOAD_PREFIX = "gpu_reload_"


def _find_all_effective_users(node: fx.Node, op_types: OpTypes) -> OrderedSet[fx.Node]:
    """Find all effective users of a node, where view ops extend the lifetime
    of the original node. If a user is a view op, recursively find users of
    the view."""
    effective_users: OrderedSet[fx.Node] = OrderedSet()
    for user in node.users:
        if user.op == "output":
            continue
        effective_users.add(user)
        if op_types.is_view(user):
            effective_users.update(_find_all_effective_users(user, op_types))
    return effective_users


@dataclass
class ReloadNodeInfo:
    """
    Information about backward reload related nodes for each reload operation.

    Pattern: ao.reload → ao.wait_tensor

    - Reload group (ao.reload): Performs the actual asynchronous data transfer.
      Can be moved earlier in the graph to overlap with computation.
    - Wait node (ao.wait_tensor): Synchronization point that blocks until the data
      transfer completes. Must remain at the point where the data is first needed.
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
    Insert CPU offload operations in the forward pass graph.

    Offload operations are placed after the last effective use of each tensor marked
    for offloading. This ensures the tensor is no longer needed on the GPU before
    transferring it to CPU memory.

    NOTE: An alternative approach would offload tensors immediately after generation
    to maximize compute-communication overlap. However, this requires additional
    synchronization to ensure tensor deletion (which occurs on the default stream)
    waits for the asynchronous offload operation to complete. This would necessitate
    more complex tracking to separate operation scheduling from memory cleanup.

    Args:
        graph: The forward graph to modify
    """

    op_types: OpTypes = get_default_op_list()

    output_node: fx.Node = graph.find_nodes(op="output")[0]
    # pyrefly: ignore [bad-assignment]
    fwd_outputs: tuple[fx.Node, ...] = output_node.args[
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
        if all_effective_users := _find_all_effective_users(node, op_types):
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


def reload_activation_bw(graph: fx.Graph) -> None:
    """
    Insert GPU reload operations in the backward pass graph.

    Reload operations are placed before the first use of each offloaded tensor,
    transferring it from CPU back to GPU memory before it's needed for computation.

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


def offload_activation_fw_async(graph: fx.Graph) -> None:
    """Insert async CPU offload operations in the forward pass graph.

    Uses ao.offload + ao.wait_tensor ops which encapsulate stream management
    internally, producing a clean 2-node IR per offloaded tensor.
    """

    op_types: OpTypes = get_default_op_list()

    output_node: fx.Node = graph.find_nodes(op="output")[0]
    # pyrefly: ignore [bad-assignment]
    fwd_outputs: tuple[fx.Node, ...] = output_node.args[
        0
    ]  # pyrefly: ignore [bad-assignment]
    node_to_offload: dict[fx.Node, fx.Node] = dict()
    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    if not any(n.meta.get("saved_for_offloading", False) for n in fwd_outputs):
        return

    for node in fwd_outputs:
        if node.meta.get("saved_for_offloading", False) is False:
            continue

        if all_effective_users := _find_all_effective_users(node, op_types):
            last_user = max(all_effective_users, key=lambda n: node_to_index[n])
        else:
            last_user: fx.Node = node

        with graph.inserting_after(last_user):
            offload_node: fx.Node = graph.call_function(
                torch.ops.ao.offload.default,
                args=(node,),
                name=f"async_{CPU_OFFLOAD_PREFIX}{node.name}",
            )
            offload_node.meta["val"] = node.meta["val"].to(torch.device("cpu"))
            offload_node.meta["tensor_meta"] = extract_tensor_metadata(
                offload_node.meta["val"]
            )
        # The keepalive=node arg extends the GPU tensor's lifetime in the
        # graph so the allocator doesn't reclaim it before the async D2H
        # copy completes.
        with graph.inserting_after(offload_node):
            wait_node: fx.Node = graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(offload_node, node),
                name=CPU_OFFLOAD_PREFIX + str(node.name),
            )
            wait_node.meta["val"] = offload_node.meta["val"]
            wait_node.meta["tensor_meta"] = offload_node.meta["tensor_meta"]

        node_to_offload[node] = wait_node

    output_node.update_arg(
        0, tuple(node_to_offload.get(node, node) for node in fwd_outputs)
    )


def reload_activation_bw_async(graph: fx.Graph) -> None:
    """Insert async GPU reload operations in the backward pass graph.

    Uses ao.reload + ao.wait_tensor ops which encapsulate stream management internally,
    producing a clean 2-node IR per reloaded tensor.
    """

    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    nodes_to_reload = [
        n
        for n in graph.find_nodes(op="placeholder")
        if n.meta.get("saved_for_offloading", False)
    ]
    if not nodes_to_reload:
        return

    for node in nodes_to_reload:
        if not node.users:
            raise RuntimeError(
                f"Offloaded tensor {node.name} has no users in the backward graph"
            )
        insert_point: fx.Node = min(node.users.keys(), key=lambda n: node_to_index[n])

        original_device: torch.device = node.meta["original_device"]
        with graph.inserting_before(insert_point):
            reload_node: fx.Node = graph.call_function(
                torch.ops.ao.reload.default,
                args=(node, original_device),
                name=f"async_{str(node.name).replace(CPU_OFFLOAD_PREFIX, GPU_RELOAD_PREFIX)}",
            )
            reload_node.meta["val"] = node.meta["val"].to(original_device)
            reload_node.meta["tensor_meta"] = extract_tensor_metadata(
                reload_node.meta["val"]
            )
            wait_node: fx.Node = graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(reload_node,),
                name=str(node.name).replace(CPU_OFFLOAD_PREFIX, GPU_RELOAD_PREFIX),
            )
            wait_node.meta["val"] = reload_node.meta["val"]
            wait_node.meta["tensor_meta"] = reload_node.meta["tensor_meta"]

        for user in list(node.users.keys()):
            if user != reload_node:
                user.replace_input_with(node, wait_node)


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

    NOTE: Additional context for the logic behind these offloading checks:

    * fwd_outputs: Only saved intermediate tensors should be offloaded.

    * model_outputs / static_lifetime_input_nodes: Tensors that may be accessed outside
      the compiled region (e.g., model outputs, static inputs) cannot be offloaded as
      they must remain accessible beyond the scope of the compiled graph.

    * views / getitems: Offloading such nodes can lead to segmentation faults.

    * contiguous: Offloading non-contiguous tensors causes CPU-side stride changes
      during both forward and backward passes when using the Inductor backend. While
      these stride changes cancel each other out, they introduce significant compute
      overhead. This is due to the contiguity check in ir.py (see link below).
      TODO: This restriction could potentially be bypassed in the future.
      Reference: https://github.com/pytorch/pytorch/blob/44ac69388a4a5eb463dbd2a13f00d1e3b924566c/torch/_inductor/ir.py#L3214

    Additional criteria to consider for offloading optimization:

    * Tensor size: Small tensors may not fully utilize available bandwidth, reducing the
      efficiency gains from offloading.

    * Position in forward/backward graph: Activations generated near the end of the forward
      pass are typically consumed near the beginning of the backward pass. Offloading such
      tensors may be counterproductive since they are quickly reloaded, not having sufficient
      time to overlap the transfer with computation.
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
    if op_types.is_view(node):
        log.debug("\tSkipped! Cannot offload views.")
        return False
    if node.target == operator.getitem:
        log.debug("\tSkipped! Cannot offload getitems.")
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
        if node.meta.get("should_offload", False) and can_offload(
            node, fwd_outputs, model_outputs, static_lifetime_input_nodes
        ):
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
    reload_activation_bw(bwd_module.graph)


def offload_chosen_sets_async(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """
    Add async offload and reload nodes using ao ops.

    Uses ao.offload/ao.reload + ao.wait_tensor which encapsulate stream management,
    instead of device_put + explicit stream operations. Can be applied to
    partitioned forward/backward graphs or to a joint graph produced by make_fx.
    """

    offload_activation_fw_async(fwd_module.graph)

    # Replace backward graph placeholders with their offloaded (CPU) counterparts.
    # For each offloaded forward output, find the matching backward input and swap
    # it with a new placeholder carrying the CPU tensor's metadata, then mark it
    # for reloading.
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

    reload_activation_bw_async(bwd_module.graph)


def activation_offload_sink_wait_async(fwd_module: fx.GraphModule) -> None:
    """Sink ao.wait_tensor operations for offload completion to the end of the graph.

    This allows computation to overlap with offload operations.

    NOTE: Sinking waits to the end delays GPU memory release of the source
    tensor (kept alive via the wait's keepalive arg) until the end of the
    compiled graph. For per-layer compile this is fine (one layer's worth of
    memory), but for full-model compile this means offloaded GPU tensors are
    not freed until the entire forward pass completes.
    """
    graph: fx.Graph = fwd_module.graph
    output_node: fx.Node = graph.find_nodes(op="output")[0]

    wait_nodes_to_sink: list[fx.Node] = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.ao.wait_tensor.default
        and isinstance(node.args[0], fx.Node)
        and node.args[0].op == "call_function"
        and node.args[0].target == torch.ops.ao.offload.default
    ]

    # prepend moves the node from its current position (no manual removal needed)
    for wait_node in wait_nodes_to_sink:
        output_node.prepend(wait_node)


def activation_reload_prefetch_async(bwd_module: fx.GraphModule) -> None:
    """
    Prefetch backward reload operations by moving ao.reload nodes earlier
    in the graph to overlap data transfer with computation, while keeping
    ao.wait_tensor at its original position.
    """
    graph: fx.Graph = bwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)

    # Identify reload + wait pairs
    reload_patterns: dict[fx.Node, ReloadNodeInfo] = {}
    for node in graph.nodes:
        if not (
            node.op == "call_function" and node.target == torch.ops.ao.reload.default
        ):
            continue
        wait_node = next(
            (u for u in node.users if u.target == torch.ops.ao.wait_tensor.default),
            None,
        )
        if wait_node is None:
            continue
        transfer_size_bytes: int = _calculate_transfer_size(node)
        transfer_time_ms: float = _estimate_transfer_time_in_ms(transfer_size_bytes)
        reload_patterns[node] = ReloadNodeInfo(
            reload_group_nodes=[node],
            wait_event_node=wait_node,
            transfer_size_bytes=transfer_size_bytes,
            transfer_time_ms=transfer_time_ms,
        )

    reorder_for_prefetch(nodes_list, reload_patterns)


def _calculate_transfer_size(device_put_node: fx.Node) -> int:
    """Calculate the size in bytes of data being transferred."""

    # ao.offload(tensor) -> tensor at args[0]
    # ao.reload(tensor, device) -> tensor at args[0]
    if device_put_node.target in (
        torch.ops.ao.offload.default,
        torch.ops.ao.reload.default,
    ):
        return _size_of(device_put_node.args[0])  # pyrefly: ignore [bad-argument-type]
    raise ValueError(f"Unexpected transfer op: {device_put_node.target}")


def _estimate_transfer_time_in_ms(transfer_size_bytes: int) -> float:
    """Estimate transfer time in milliseconds based on size and bandwidth.

    Uses config.activation_offload_cpu_gpu_bw (GB/s) which should be set by
    the user to match their hardware.
    """
    return (
        transfer_size_bytes / (1024**3) * 1_000 / config.activation_offload_cpu_gpu_bw
    )


def identify_reload_patterns(
    graph: fx.Graph, nodes_list: list[fx.Node], node_to_idx: dict[fx.Node, int]
) -> dict[fx.Node, ReloadNodeInfo]:
    """
    Identify backward reload patterns in the graph.

    Pattern: fork → wait_stream → device_put → record_event → join → wait_event

    This uses position-based matching since these nodes are inserted together in
    add_backward_reload_stream_ops() in a specific order. Since stream operations
    do not have data dependencies between them, they are unsuitable for subgroup
    pattern matching type of checks.

    Returns a dict mapping device_put node to ReloadNodeInfo containing:
    - reload_group_nodes: fork → wait_stream → device_put → record_event → join
    - wait_event_node: the wait_event node
    - transfer_size_bytes: size of data being transferred
    - transfer_time_ms: estimated transfer time in milliseconds
    """
    patterns: dict[fx.Node, ReloadNodeInfo] = {}

    # Find all GPU reload device_put nodes whose inputs are placeholder nodes
    reload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if GPU_RELOAD_PREFIX in node.name
        and (
            node.args
            and isinstance(node.args[0], fx.Node)
            and node.args[0].op == "placeholder"
        )
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
        # Removed in follow-up commit
        _validate_pattern_nodes(  # noqa: F821  # pyrefly: ignore [unknown-name]
            fork_node,
            wait_stream_node,
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
    reorder_for_prefetch(nodes_list, reload_patterns)


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
        num_fwd_outputs,
        static_lifetime_input_nodes,
    )
    if not should_perform_offloading:
        return

    # Step 2: Add offload and reload nodes to the graphs
    if config.activation_offload_separate_stream:
        # Use async ao ops (2 nodes each: offload/reload + wait_tensor)
        offload_chosen_sets_async(fwd_module, bwd_module)
        if config.activation_offload_sink_wait:
            activation_offload_sink_wait_async(fwd_module)
        if config.activation_reload_prefetch:
            activation_reload_prefetch_async(bwd_module)
    else:
        # Use synchronous device_put (1 node each)
        offload_chosen_sets(fwd_module, bwd_module)

    fwd_module.graph.lint()
    bwd_module.graph.lint()
