"""
Activation offloading for memory optimization in (more like post) partitioners.

This module provides functionality to offload activations to CPU during forward pass
and reload them during backward pass, reducing GPU memory usage.
"""

import logging
import operator

import torch
import torch.fx as fx
from torch._dynamo.variables.streams import get_current_stream, new_event, new_stream
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..partitioners import get_default_op_list, OpTypes


log: logging.Logger = logging.getLogger(__name__)


# Node name prefixes for offload/reload operations
CPU_OFFLOAD_PREFIX = "cpu_offload_"
GPU_RELOAD_PREFIX = "gpu_reload_"


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

    fwd_module.graph.lint()
    bwd_module.graph.lint()
