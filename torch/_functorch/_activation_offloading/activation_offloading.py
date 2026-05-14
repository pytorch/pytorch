"""
Activation offloading for memory optimization in (more like post) partitioners.

This module provides functionality to offload activations to CPU during forward pass
and reload them during backward pass, reducing GPU memory usage.

Additional TODO:
* given the fact that PT2 stream support is in active development, testings should
  be done once that is more finalized. A issue currently known is that with streams,
  each iteration will have its own offload streams, but the streams should be shared
  across the iterations.
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
# NOTE: right now we are using these prefixes as identifiers for offload/reload
CPU_OFFLOAD_PREFIX = "cpu_offload_"
GPU_RELOAD_PREFIX = "gpu_reload_"


@dataclass
class ReloadNodeInfo:
    """
    Information about backward reload related nodes for each reload operation.

    Pattern: device_put [stream metadata] → record_event → ... → wait_event

    This pattern is divided into two logical groups for optimization purposes:
    - Reload group (device_put → record_event):
      Performs the actual asynchronous data transfer on a separate stream
      (identified by node.meta["custom"]["stream"]).
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

    def find_all_effective_users(node: fx.Node) -> OrderedSet[fx.Node]:
        """
        Find all effective users of a node, where view ops extend the lifetime of the
        original node. If a user is a view op, recursively find users of the view.
        """
        effective_users: OrderedSet[fx.Node] = OrderedSet()
        for user in node.users:
            if user.op == "output":
                continue
            effective_users.add(user)
            if op_types.is_view(user):
                effective_users.update(find_all_effective_users(user))

        return effective_users

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


def _find_hook_offload_nodes(graph: fx.Graph) -> list[fx.Node]:
    """Find prims.device_put nodes that copy GPU→CPU (from inlined hooks)."""
    return [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.prims.device_put.default
        and isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].device.type == "cpu"
        and isinstance(node.args[0], fx.Node)
        and isinstance(node.args[0].meta.get("val"), torch.Tensor)
        and node.args[0].meta["val"].device.type == "cuda"
        and CPU_OFFLOAD_PREFIX not in node.name
    ]


def _find_hook_reload_nodes(graph: fx.Graph) -> list[fx.Node]:
    """Find prims.device_put nodes that copy CPU→GPU (from inlined hooks)."""
    return [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.prims.device_put.default
        and isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].device.type == "cuda"
        and isinstance(node.args[0], fx.Node)
        and isinstance(node.args[0].meta.get("val"), torch.Tensor)
        and node.args[0].meta["val"].device.type == "cpu"
        and GPU_RELOAD_PREFIX not in node.name
    ]


def put_offload_nodes_on_separate_stream(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """
    Annotate offload/reload device_put nodes with stream metadata and sync events.

    Args:
        fwd_module: Forward module graph
        bwd_module: Backward module graph
    """
    fwd_offload_nodes = [
        node
        for node in fwd_module.graph.nodes
        if CPU_OFFLOAD_PREFIX in node.name and node.op == "call_function"
    ]
    bwd_reload_nodes = [
        node
        for node in bwd_module.graph.nodes
        if GPU_RELOAD_PREFIX in node.name and node.op == "call_function"
    ]

    _annotate_forward_offload_stream(fwd_module.graph, fwd_offload_nodes)
    _annotate_backward_reload_stream(bwd_module.graph, bwd_reload_nodes)


def _hoist_offload_device_puts(
    graph: fx.Graph,
    offload_nodes: list[fx.Node],
) -> None:
    """Move device_put nodes earlier to enable overlap with compute.

    After hook inlining, device_put nodes for CPU offloading end up at the
    return boundary of the forward graph — after ALL compute. This prevents
    overlap because there's no compute running concurrently with the D2H copy.

    This function moves each device_put to right after the last forward use
    of its input tensor. Combined with sink_wait, this creates an overlap
    window: the D2H copy on the offload stream runs concurrently with
    subsequent forward compute on the default stream.
    """
    node_to_idx: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    for dp_node in offload_nodes:
        tensor_node: fx.Node = dp_node.args[0]  # type: ignore[assignment]
        if not isinstance(tensor_node, fx.Node):
            continue

        # Find the last use of tensor_node that isn't the device_put itself
        # or the output node.
        last_use: fx.Node | None = None
        last_use_idx: int = -1
        for user in tensor_node.users:
            if user is dp_node or user.op == "output":
                continue
            user_idx = node_to_idx.get(user, -1)
            if user_idx > last_use_idx:
                last_use = user
                last_use_idx = user_idx

        if last_use is not None:
            last_use.append(dp_node)
            # Update index for moved node
            node_to_idx[dp_node] = last_use_idx + 1


def _annotate_forward_offload_stream(
    graph: fx.Graph,
    offload_nodes: list[fx.Node],
) -> None:
    """Tag forward offload device_put nodes with stream metadata and sync events.

    Uses node.meta["custom"]["stream"] for stream assignment (same mechanism as
    torch.cuda.stream() in dynamo) plus record_event/wait_event for cross-stream
    synchronization and record_stream for CUDA allocator memory safety.

    Pattern per device_put:
      record_event(ready, default) → wait_event(ready, offload) →
      record_stream(tensor, offload) → device_put [metadata: offload stream] →
      record_event(done, offload) → wait_event(done, default) →
      record_stream(tensor, default)
    """
    if not offload_nodes:
        return

    current_stream_id: int = get_current_stream(
        offload_nodes[0].args[0].meta["val"].device  # type: ignore[assignment]
    )
    offload_stream_id: int = new_stream()

    for offload_node in offload_nodes:
        ready_event_id: int = new_event()
        completion_event_id: int = new_event()
        tensor_node: fx.Node = offload_node.args[0]  # type: ignore[assignment]

        offload_node.meta.setdefault("custom", {})["stream"] = offload_stream_id

        with graph.inserting_before(offload_node):
            graph.call_function(
                torch.ops.streams.record_event.default,
                args=(ready_event_id, current_stream_id),
            )
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(ready_event_id, offload_stream_id),
            )
            graph.call_function(
                torch.ops.streams.record_stream.default,
                args=(tensor_node, offload_stream_id),
                name=f"record_stream_{tensor_node.name}",
            )

        with graph.inserting_after(offload_node):
            record_node = graph.call_function(
                torch.ops.streams.record_event.default,
                args=(completion_event_id, offload_stream_id),
            )
        with graph.inserting_after(record_node):
            wait_node = graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(completion_event_id, current_stream_id),
            )
        with graph.inserting_after(wait_node):
            graph.call_function(
                torch.ops.streams.record_stream.default,
                args=(tensor_node, current_stream_id),
                name=f"keep_alive_{tensor_node.name}",
            )


def _annotate_backward_reload_stream(
    graph: fx.Graph,
    reload_nodes: list[fx.Node],
    device: torch.device | None = None,
) -> None:
    """Tag backward reload device_put nodes with stream metadata and sync events.

    Pattern per device_put:
      device_put [metadata: reload stream] → record_event(done, reload) →
      ... (other backward compute) ...
      wait_event(done, default)  [placed at first use of the reloaded tensor]
    """
    if not reload_nodes:
        return

    if device is None:
        device = reload_nodes[0].args[0].meta["original_device"]
    current_stream_id: int = get_current_stream(device)
    reload_stream_id: int = new_stream()

    node_to_index: dict[fx.Node, int] = {
        node: idx for idx, node in enumerate(graph.nodes)
    }

    for reload_node in reload_nodes:
        event_id: int = new_event()

        reload_node.meta.setdefault("custom", {})["stream"] = reload_stream_id

        with graph.inserting_after(reload_node):
            record_node = graph.call_function(
                torch.ops.streams.record_event.default,
                args=(event_id, reload_stream_id),
            )

        # Place wait_event at the first USE of the reload result so backward
        # compute that doesn't need this tensor overlaps with the H2D copy.
        first_user = min(
            (u for u in reload_node.users if u != record_node),
            key=lambda n: node_to_index.get(n, float("inf")),
            default=None,
        )
        insert_point = first_user if first_user is not None else record_node

        with graph.inserting_before(insert_point):
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(event_id, current_stream_id),
            )


def put_hook_offload_on_separate_stream(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """Annotate hook-inlined offload/reload device_put nodes with stream metadata.

    Called after maybe_inline_graph_saved_tensors_hooks in graph_compile.py.
    """
    fwd_offload_nodes = _find_hook_offload_nodes(fwd_module.graph)
    bwd_reload_nodes = _find_hook_reload_nodes(bwd_module.graph)

    if not fwd_offload_nodes and not bwd_reload_nodes:
        return

    # Move device_put nodes earlier so they're interleaved with compute,
    # enabling overlap between D2H copies and subsequent forward compute.
    if fwd_offload_nodes:
        _hoist_offload_device_puts(fwd_module.graph, fwd_offload_nodes)

    _annotate_forward_offload_stream(fwd_module.graph, fwd_offload_nodes)

    if config.activation_offload_sink_wait:
        activation_offload_sink_wait(fwd_module)

    reload_device = (
        bwd_reload_nodes[0].meta["val"].device if bwd_reload_nodes else None
    )
    _annotate_backward_reload_stream(
        bwd_module.graph, bwd_reload_nodes, device=reload_device
    )

    fwd_module.recompile()
    bwd_module.recompile()


def _calculate_transfer_size(device_put_node: fx.Node) -> int:
    """Calculate the size in bytes of data being transferred."""

    return _size_of(device_put_node.args[0])  # pyrefly: ignore [bad-argument-type]


def _estimate_transfer_time_in_ms(transfer_size_bytes: int) -> float:
    """
    Estimate transfer time in milliseconds based on size and bandwidth.
    NOTE: potentially could be standardized in node estimator class
    """

    return transfer_size_bytes / (1024**3) * 1_000 / inductor_config.cpu_gpu_bw


def identify_reload_patterns(
    graph: fx.Graph, nodes_list: list[fx.Node], node_to_idx: dict[fx.Node, int]
) -> dict[fx.Node, ReloadNodeInfo]:
    """
    Identify backward reload patterns in the graph.

    Pattern: device_put [stream metadata] → record_event → ... → wait_event

    Reload device_put nodes are identified by their stream metadata
    (node.meta["custom"]["stream"]), set by _annotate_backward_reload_stream.
    The record_event immediately follows the device_put, while the wait_event
    is placed at the first use point.

    Returns a dict mapping device_put node to ReloadNodeInfo containing:
    - reload_group_nodes: [device_put, record_event]
    - wait_event_node: the wait_event node (at first use point)
    - transfer_size_bytes: size of data being transferred
    - transfer_time_ms: estimated transfer time in milliseconds
    """
    patterns: dict[fx.Node, ReloadNodeInfo] = {}

    # Find all GPU reload device_put nodes with stream metadata set.
    reload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if node.args
        and isinstance(node.args[0], fx.Node)
        and node.meta.get("custom", {}).get("stream") is not None
        and (
            GPU_RELOAD_PREFIX in node.name
            or (
                isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].device.type == "cuda"
                and isinstance(node.args[0].meta.get("val"), torch.Tensor)
                and node.args[0].meta["val"].device.type == "cpu"
            )
        )
    ]

    for reload_node in reload_nodes:
        reload_node_idx: int = node_to_idx[reload_node]

        # record_event is placed immediately after device_put
        record_event_node: fx.Node = nodes_list[reload_node_idx + 1]
        if not (
            record_event_node.op == "call_function"
            and record_event_node.target == torch.ops.streams.record_event.default
        ):
            continue

        # wait_event is placed at first use point — find it by matching
        # the event_id from record_event
        event_id = record_event_node.args[0]
        wait_event_node: fx.Node | None = None
        for node in nodes_list[reload_node_idx + 2 :]:
            if (
                node.op == "call_function"
                and node.target == torch.ops.streams.wait_event.default
                and node.args[0] == event_id
            ):
                wait_event_node = node
                break

        if wait_event_node is None:
            continue

        transfer_size_bytes: int = _calculate_transfer_size(reload_node)
        transfer_time_ms: float = _estimate_transfer_time_in_ms(transfer_size_bytes)

        patterns[reload_node] = ReloadNodeInfo(
            reload_group_nodes=[reload_node, record_event_node],
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
    Sink forward wait_event + keep_alive record_stream to the end of the
    graph so the compute stream overlaps with D2H copies but still
    synchronizes before returning.

    Without sinking, the wait_event immediately after each device_put
    forces the compute stream to block on every D2H copy. By moving it
    to just before the output node, compute proceeds unblocked while copies
    run on the offload stream. The final wait ensures all copies complete
    before the forward outputs are consumed by backward.

    The keep_alive record_stream must be sunk alongside the wait_event
    because it is the last graph reference to the source tensor. If it
    stays at the original offload site, the inductor memory planner sees
    the tensor as dead and may reuse its buffer for subsequent compute
    while the D2H copy is still reading from it.
    """
    graph: fx.Graph = fwd_module.graph
    nodes_list: list[fx.Node] = list(graph.nodes)
    node_to_idx: dict[fx.Node, int] = {node: idx for idx, node in enumerate(nodes_list)}

    output_node = next(n for n in graph.nodes if n.op == "output")

    offload_nodes: list[fx.Node] = [
        node
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.prims.device_put.default
        )
        if (
            CPU_OFFLOAD_PREFIX in node.name
            or (
                isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].device.type == "cpu"
                and isinstance(node.args[0], fx.Node)
                and isinstance(node.args[0].meta.get("val"), torch.Tensor)
                and node.args[0].meta["val"].device.type == "cuda"
            )
        )
    ]

    for offload_node in offload_nodes:
        offload_idx: int = node_to_idx[offload_node]

        # Pattern: ... → device_put → record_event → wait_event → keep_alive
        # Sink wait_event and keep_alive (record_stream) to just before the
        # output node.  The keep_alive extends cos_1's liveness so the memory
        # planner won't reuse its buffer while the D2H copy is in flight.
        nodes_to_sink = []
        for offset in [1, 2, 3]:
            if offload_idx + offset >= len(nodes_list):
                break
            candidate = nodes_list[offload_idx + offset]
            if candidate.target in (
                torch.ops.streams.wait_event.default,
                torch.ops.streams.record_stream.default,
            ):
                nodes_to_sink.append(candidate)

        for node in nodes_to_sink:
            target = node.target
            args = node.args
            meta = node.meta
            node.replace_all_uses_with(None)
            graph.erase_node(node)
            with graph.inserting_before(output_node):
                new_node = graph.call_function(target, args=args)
                new_node.meta = meta


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
    offload_chosen_sets(fwd_module, bwd_module)

    # Step 3: Put offload nodes on separate stream if configured
    if config.activation_offload_separate_stream:
        put_offload_nodes_on_separate_stream(fwd_module, bwd_module)
        if config.activation_offload_sink_wait:
            activation_offload_sink_wait(fwd_module)
        if config.activation_reload_prefetch:
            activation_reload_prefetch(bwd_module)

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )

        wrap_all_sync_nodes_with_control_deps(fwd_module)
        wrap_all_sync_nodes_with_control_deps(bwd_module)

    fwd_module.graph.lint()
    bwd_module.graph.lint()
