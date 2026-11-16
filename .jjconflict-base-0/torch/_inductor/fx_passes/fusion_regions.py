"""Detect fusion regions for overlap scheduling."""

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


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


def estimate_fused_node_costs(
    region_of: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, float]:
    """Estimate per-node costs accounting for fusion (no graph collapse).

    For each node in a fusion region, only count I/O that crosses the
    region boundary (external reads/writes). Internal intermediates
    won't be materialized by inductor, so they shouldn't count.

    Nodes with only internal I/O get cost 0. Nodes with external I/O
    get the bandwidth cost of just those external tensors.
    """
    from torch.utils._pytree import tree_flatten
    from torch.utils._runtime_estimation import get_transfer_time

    costs: dict[fx.Node, float] = {}
    seen: OrderedSet[int] = OrderedSet()

    for node_set in region_of.values():
        rid = id(node_set)
        if rid in seen:
            continue
        seen.add(rid)

        for n in node_set:
            # External inputs: values from nodes outside this region
            ext_inputs = []
            for inp in n.all_input_nodes:
                if inp not in node_set:
                    val = inp.meta.get("val")
                    if val is not None:
                        ext_inputs.append(val)

            # External outputs: this node's value is used outside the region
            ext_outputs = []
            has_external_user = any(u not in node_set for u in n.users)
            if has_external_user:
                val = n.meta.get("val")
                if val is not None:
                    ext_outputs.append(val)

            if not ext_inputs and not ext_outputs:
                costs[n] = 0.0
            else:
                flat_in, _ = tree_flatten(ext_inputs)
                flat_out, _ = tree_flatten(ext_outputs)
                costs[n] = get_transfer_time(flat_in, flat_out) / 1e6

    return costs
