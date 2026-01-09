"""Detect fusion regions for overlap scheduling."""

import operator
from dataclasses import dataclass, field

import torch
import torch.fx as fx
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet
from torch.utils._runtime_estimation import get_num_bytes


@dataclass
class FusionRegion:
    """Represents a connected set of fusible operations that will fuse together."""

    subgraph_node: fx.Node  # The call_module node for this fusion
    subgraph_module: fx.GraphModule  # The subgraph module
    total_bytes: int = field(default=0, init=False)  # Total input + output bytes
    cost_ms: float = field(default=0.0, init=False)  # Estimated cost in milliseconds

    def __post_init__(self) -> None:
        """Compute cost based on subgraph's placeholder inputs and output node."""
        self.total_bytes, self.cost_ms = self._compute_cost()

    def _compute_cost(self) -> tuple[int, float]:
        from torch.utils._pytree import tree_flatten
        from torch.utils._runtime_estimation import get_transfer_time

        subgraph = self.subgraph_module
        input_vals = [
            n.meta.get("val") for n in subgraph.graph.find_nodes(op="placeholder")
        ]
        output_vals = [
            n.meta.get("val")
            for n in torch._inductor.utils.output_node(subgraph).all_input_nodes
        ]
        flat_inputs, _ = tree_flatten(input_vals)
        flat_outputs, _ = tree_flatten(output_vals)

        transfer_time_ns = get_transfer_time(flat_inputs, flat_outputs)
        total_bytes = sum(
            get_num_bytes(t)
            for t in flat_inputs + flat_outputs
            if isinstance(t, torch.Tensor)
        )
        return total_bytes, transfer_time_ns / 1e6


def is_view_node(n: fx.Node) -> bool:
    """Check if a node is a view operation (zero cost, no memory allocation)."""

    return isinstance(n.target, torch._ops.OpOverload) and (
        n.target.is_view and n.target.namespace in ("aten", "prims")
    )


def is_fusible_node(n: fx.Node) -> bool:
    """Check if a node is fusible based on whether it has an inductor lowering.

    A node is fusible if:
    - It has a lowering in torch._inductor.lowering.lowerings
    - It does NOT have a flop counter (expensive compute ops like mm/conv)
    - It is NOT a registered fallback (ops that fall back to eager)
    - It is NOT a collective or wait op
    - For aten.cat, it must have <= max_pointwise_cat_inputs inputs
    """
    if n.op != "call_function":
        return False

    target = n.target
    if not isinstance(target, torch._ops.OpOverload):
        return False

    # Exclude collectives and waits (they have their own scheduling)
    if target.namespace == "_c10d_functional":
        return False

    from torch._inductor.lowering import fallbacks, lowerings
    from torch.utils.flop_counter import flop_registry

    # Must have a lowering
    if target not in lowerings:
        return False

    # Exclude fallbacks (ops that fall back to eager execution)
    if target in fallbacks:
        return False

    # Exclude ops with flop counters (expensive compute ops like mm, conv, etc.)
    overload_packet = target.overloadpacket
    if overload_packet in flop_registry:
        return False

    # Special case: cat is only fusible if it has few enough inputs
    if target == torch.ops.aten.cat.default:
        inputs = n.args[0] if n.args else []
        if isinstance(inputs, (list, tuple)):
            import torch._inductor.config as inductor_config

            if len(inputs) > inductor_config.max_pointwise_cat_inputs:
                return False

    return True


def _get_contiguous_fusible_spans(gm: fx.GraphModule) -> list[list[fx.Node]]:
    """Get contiguous spans of fusible nodes from the graph.

    Walks the graph in topological order and groups consecutive fusible
    nodes into spans. Non-fusible nodes act as span boundaries.
    """
    spans: list[list[fx.Node]] = []
    current_span: list[fx.Node] = []

    for node in gm.graph.nodes:
        if is_fusible_node(node):
            current_span.append(node)
        else:
            # Non-fusible node ends the current span
            if current_span:
                spans.append(current_span)
                current_span = []

    if current_span:
        spans.append(current_span)

    return spans


def _find_connected_components(span: list[fx.Node]) -> list[list[fx.Node]]:
    """Find connected components within a span of fusible nodes.

    Two nodes are connected if one is an input to the other (direct data dependency).
    """
    if not span:
        return []

    from torch.fx.experimental.optimization import UnionFind

    span_set = OrderedSet(span)
    node_to_idx = {n: i for i, n in enumerate(span)}

    uf = UnionFind(len(span))
    for i in range(len(span)):
        uf.make_set(i)

    # Union nodes based on input edges
    for node in span:
        node_idx = node_to_idx[node]
        for inp in node.all_input_nodes:
            if inp in span_set:
                uf.join(node_idx, node_to_idx[inp])

    # Group by root
    root_to_nodes: dict[int, list[fx.Node]] = {}
    for node in span:
        root = uf.find(node_to_idx[node])
        if root not in root_to_nodes:
            root_to_nodes[root] = []
        root_to_nodes[root].append(node)

    return list(root_to_nodes.values())


def build_fusion_regions(
    gm: fx.GraphModule,
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """Build fusion regions from contiguous spans of fusible nodes.

    1. Identify contiguous spans of fusible nodes (separated by non-fusible nodes)
    2. Find connected components within each span
    3. Return regions that have 2+ non-view nodes

    This ensures fusion regions are strictly local - no reordering across
    non-fusible node boundaries.

    Returns a dict mapping each node to its fusion group (OrderedSet of nodes).
    """
    # Build node -> topo index map for sorting
    node_to_idx: dict[fx.Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # Step 1: Get contiguous spans of fusible nodes
    spans = _get_contiguous_fusible_spans(gm)

    # Step 2: Find connected components within each span
    region_of: dict[fx.Node, OrderedSet[fx.Node]] = {}

    for span in spans:
        if len(span) < 2:
            continue

        components = _find_connected_components(span)

        for component in components:
            # Skip regions with fewer than 2 non-view nodes (views have no cost)
            non_view_count = sum(1 for n in component if not is_view_node(n))
            if non_view_count < 2:
                continue

            # Sort nodes in topological order to preserve original ordering
            sorted_component = sorted(component, key=lambda n: node_to_idx[n])
            node_set = OrderedSet(sorted_component)

            for node in sorted_component:
                region_of[node] = node_set

    return region_of


def collapse_fusion_regions(
    gm: fx.GraphModule,
    region_of: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, FusionRegion]:
    """
    Collapse fusion regions into call_module nodes using fuse_by_partitions.
    Returns new_region_of mapping module nodes to FusionRegions.
    """
    from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

    if not region_of:
        return {}

    # Get unique node sets (regions with <2 nodes already filtered in build_fusion_regions)
    unique_regions: list[tuple[OrderedSet[fx.Node], int]] = []
    seen_region_ids: OrderedSet[int] = OrderedSet()
    for node_set in region_of.values():
        region_id = id(node_set)
        if region_id not in seen_region_ids:
            seen_region_ids.add(region_id)
            unique_regions.append((node_set, region_id))

    if not unique_regions:
        return {}

    # Log graph before fusion
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fusion_regions_before",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(print_output=False),
    )

    # Build partitions list for fuse_by_partitions
    partitions = [dict.fromkeys(nodes) for nodes, _ in unique_regions]

    # Fuse all partitions at once
    fuse_by_partitions(gm, partitions, prefix="_fusion_region_")

    # Log graph after fusion
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fusion_regions_after",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(print_output=False),
    )

    # Build new_region_of by finding the call_module nodes
    new_region_of: dict[fx.Node, FusionRegion] = {}

    for region_idx in range(len(unique_regions)):
        subgraph_name = f"_fusion_region_{region_idx}"

        # Find the call_module node
        module_nodes = list(gm.graph.find_nodes(op="call_module", target=subgraph_name))
        assert len(module_nodes) == 1, (
            f"Expected 1 call_module for {subgraph_name}, got {len(module_nodes)}"
        )
        module_node = module_nodes[0]

        subgraph_module = getattr(gm, subgraph_name)

        # Create FusionRegion with all required info
        region = FusionRegion(
            subgraph_node=module_node,
            subgraph_module=subgraph_module,
        )

        new_region_of[module_node] = region

    return new_region_of


def expand_fusion_regions(
    gm: fx.GraphModule,
    region_of: dict[fx.Node, FusionRegion],
) -> dict[fx.Node, fx.Node | None]:
    """
    Expand call_module nodes back to their original nodes using _inline_module.

    Returns a mapping from erased module nodes to their replacement (last inlined node).
    This is used with transfer_erased_node_deps to update dependencies.
    """
    from torch.fx.experimental.const_fold import _inline_module

    result: dict[fx.Node, fx.Node | None] = {}

    if not region_of:
        return result

    for module_node, region in list(region_of.items()):
        if module_node.op != "call_module":
            continue

        subgraph_name = module_node.target
        assert isinstance(subgraph_name, str)
        assert hasattr(gm, subgraph_name), (
            f"Expected submodule {subgraph_name} to exist"
        )

        # Users of module_node are get_items that will be removed from the graph
        for user in module_node.users:
            if user.op == "call_function" and user.target == operator.getitem:
                result[user] = None

        # Get the output arg from the subgraph to determine what will replace module_node
        output_arg = torch._inductor.utils.output_node(region.subgraph_module).args[0]

        # Inline the module and get the mapping from subgraph nodes to new nodes
        subgraph_to_new = _inline_module(gm, subgraph_name)

        # Map module_node to the replacement for the output arg
        # For multi-output (tuple), use the last element (latest in topo order)
        # so dependencies are only satisfied after all outputs are computed
        if isinstance(output_arg, (list, tuple)):
            if output_arg:
                last_arg = output_arg[-1]
                assert isinstance(last_arg, fx.Node)
                result[module_node] = subgraph_to_new[last_arg]
        elif isinstance(output_arg, fx.Node) and output_arg in subgraph_to_new:
            result[module_node] = subgraph_to_new[output_arg]

        delattr(gm, subgraph_name)

    return result
