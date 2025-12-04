"""Detect fusion regions for overlap scheduling."""

import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


@dataclass
class FusionRegion:
    """Represents a connected set of fusible operations that will fuse together."""

    nodes: OrderedSet[fx.Node]  # All nodes in topo order
    cost_ms: float = 0.0  # Estimated cost in milliseconds
    external_inputs: OrderedSet[fx.Node] = None  # Inputs from outside the region
    external_outputs: OrderedSet[fx.Node] = None  # Nodes with users outside the region
    external_users: OrderedSet[fx.Node] = None  # Users outside the region
    subgraph_node: fx.Node | None = None  # The subgraph node representing this region

    def __post_init__(self):
        """Compute cost and external inputs/outputs."""
        from torch._inductor.utils import get_gpu_dram_gbps

        region_set = set(self.nodes)
        self.external_inputs = OrderedSet()
        self.external_outputs = OrderedSet()
        self.external_users = OrderedSet()

        for node in self.nodes:
            # Collect all external inputs (not just tensors)
            for inp in node.all_input_nodes:
                if inp not in region_set:
                    self.external_inputs.add(inp)

            # Collect all external outputs (not just tensors)
            if any(u not in region_set for u in node.users) or len(node.users) == 0:
                self.external_outputs.add(node)

            # Collect all external users
            for user in node.users:
                if user not in region_set:
                    self.external_users.add(user)

        # Calculate cost from tensor metadata of external IO
        total_bytes = 0
        for node in self.external_inputs | self.external_outputs:
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue

            total_bytes += val.numel() * val.element_size()

        if total_bytes > 0:
            fusion_bw_gbps = get_gpu_dram_gbps()
            fusion_bw_bytes_per_s = fusion_bw_gbps * 1e9
            self.cost_ms = (total_bytes / fusion_bw_bytes_per_s) * 1000

    @property
    def start(self) -> fx.Node:
        """First node in the region."""
        return next(iter(self.nodes))

    @property
    def end(self) -> fx.Node:
        """Last node (anchor) in the region."""
        return list(self.nodes)[-1]


def is_fusible_node(n: fx.Node) -> bool:
    """Check if a node is fusible (pointwise, reduction, views, indexing ops).

    Excludes: mm/conv, collectives, waits, placeholders, outputs.
    """
    # Include pointwise, reduction, views
    tags = getattr(n.target, "tags", ())
    if torch.Tag.pointwise in tags or torch.Tag.reduction in tags:
        return True

    if getattr(n.target, "is_view", False):
        return True

    # Include specific indexing ops
    aten = torch.ops.aten
    if n.target in (aten.slice.Tensor, aten.gather.default, aten.embedding.default):
        return True

    return False


def build_fusion_regions(
    graph_nodes: list[fx.Node],
) -> dict[fx.Node, FusionRegion]:
    """Build fusion regions from the graph.

    Returns a dict mapping each node to its containing region (if any).

    Algorithm:
    1. Split graph into segments separated by non-fusible nodes
    2. Within each segment, group connected nodes via data dependencies
    """
    # Find segments: consecutive fusible nodes separated by non-fusible nodes
    segments: list[list[fx.Node]] = []
    current_segment: list[fx.Node] = []

    for node in graph_nodes:
        if is_fusible_node(node):
            current_segment.append(node)
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []

    if current_segment:
        segments.append(current_segment)

    # Build fusion regions within each segment
    region_of: dict[fx.Node, FusionRegion] = {}

    for segment in segments:
        if len(segment) < 2:
            continue

        segment_set = set(segment)
        # Map each node to its region members (initially just itself)
        node_to_region: dict[fx.Node, OrderedSet[fx.Node]] = {}
        for n in segment:
            node_to_region[n] = OrderedSet([n])

        # Build adjacency mapping for shared inputs
        input_to_consumers: dict[fx.Node, list[fx.Node]] = {}
        for node in segment:
            for inp in node.all_input_nodes:
                if inp not in segment_set:  # External input
                    if inp not in input_to_consumers:
                        input_to_consumers[inp] = []
                    input_to_consumers[inp].append(node)

        # First, merge nodes that share the same external inputs
        for inp, consumers in input_to_consumers.items():
            if len(consumers) > 1:  # Multiple consumers of the same input
                first_region = node_to_region[consumers[0]]
                for consumer in consumers[1:]:
                    consumer_region = node_to_region[consumer]
                    if first_region is not consumer_region:
                        # Merge smaller into larger
                        if len(first_region) < len(consumer_region):
                            smaller, larger = first_region, consumer_region
                        else:
                            smaller, larger = consumer_region, first_region

                        larger |= smaller
                        for n in smaller:
                            node_to_region[n] = larger
                        first_region = larger

        # Second, merge producer-consumer pairs within the segment
        for node in segment:
            fusible_inputs = [
                inp for inp in node.all_input_nodes if inp in segment_set
            ]

            for inp in fusible_inputs:
                # Merge regions (union by size)
                node_region = node_to_region[node]
                inp_region = node_to_region[inp]
                if node_region is not inp_region:
                    # Merge smaller into larger
                    if len(node_region) < len(inp_region):
                        smaller, larger = node_region, inp_region
                    else:
                        smaller, larger = inp_region, node_region

                    larger |= smaller
                    for n in smaller:
                        node_to_region[n] = larger

        # Extract unique regions
        seen_regions: set[int] = set()
        for node in segment:
            region_set = node_to_region[node]
            region_id = id(region_set)
            if region_id in seen_regions:
                continue
            seen_regions.add(region_id)

            members = list(region_set)
            if len(members) < 2:
                continue

            # Topologically sort members based on their dependencies
            members_sorted = _topological_sort_region(members)

            region = FusionRegion(nodes=OrderedSet(members_sorted))
            if region.cost_ms > 0:
                # Map all nodes to this region
                for n in members_sorted:
                    region_of[n] = region

    return region_of


def _topological_sort_region(nodes: list[fx.Node]) -> list[fx.Node]:
    """
    Topologically sort nodes within a region with sub-group awareness.

    Strategy:
    1. Identify sub-groups of nodes that share external inputs
    2. Sort within each sub-group by dependency order
    3. Sort sub-groups topologically relative to each other
    """
    node_set = set(nodes)

    # First, identify sub-groups based on shared external inputs
    subgroups = _identify_subgroups_in_region(nodes, node_set)

    # Sort each sub-group internally
    sorted_subgroups = []
    for subgroup in subgroups:
        sorted_subgroup = _sort_subgroup(subgroup, node_set)
        sorted_subgroups.append(sorted_subgroup)

    # Sort sub-groups relative to each other topologically
    return _sort_subgroups_topologically(sorted_subgroups, node_set)


def _identify_subgroups_in_region(nodes: list[fx.Node], node_set: set[fx.Node]) -> list[list[fx.Node]]:
    """Identify sub-groups within a region based on shared external inputs."""
    # Group nodes by their external input signature
    input_signatures: dict[tuple, list[fx.Node]] = {}

    for node in nodes:
        external_inputs = tuple(sorted(
            inp.name for inp in node.all_input_nodes
            if inp not in node_set
        ))

        if external_inputs not in input_signatures:
            input_signatures[external_inputs] = []
        input_signatures[external_inputs].append(node)

    return list(input_signatures.values())


def _sort_subgroup(subgroup: list[fx.Node], node_set: set[fx.Node]) -> list[fx.Node]:
    """Sort nodes within a sub-group topologically."""
    if len(subgroup) <= 1:
        return subgroup

    subgroup_set = set(subgroup)
    in_degree = {n: 0 for n in subgroup}

    # Calculate in-degrees within the sub-group
    for node in subgroup:
        for inp in node.all_input_nodes:
            if inp in subgroup_set:
                in_degree[node] += 1

    # Kahn's algorithm for the sub-group
    queue = [n for n in subgroup if in_degree[n] == 0]
    result = []

    while queue:
        queue.sort(key=lambda n: n.name)  # Deterministic ordering
        node = queue.pop(0)
        result.append(node)

        for user in node.users:
            if user in subgroup_set:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)

    return result if len(result) == len(subgroup) else subgroup


def _sort_subgroups_topologically(sorted_subgroups: list[list[fx.Node]], node_set: set[fx.Node]) -> list[fx.Node]:
    """Sort sub-groups relative to each other based on inter-subgroup dependencies."""
    if len(sorted_subgroups) <= 1:
        return sum(sorted_subgroups, [])  # Flatten single or empty list

    # Build dependency graph between sub-groups
    subgroup_deps: dict[int, set[int]] = {i: set() for i in range(len(sorted_subgroups))}

    for i, subgroup_i in enumerate(sorted_subgroups):
        for j, subgroup_j in enumerate(sorted_subgroups):
            if i != j:
                # Check if any node in subgroup_i depends on any node in subgroup_j
                for node_i in subgroup_i:
                    for inp in node_i.all_input_nodes:
                        if inp in node_set and any(inp == node_j for node_j in subgroup_j):
                            subgroup_deps[i].add(j)
                            break

    # Topologically sort sub-groups
    in_degree = {i: 0 for i in range(len(sorted_subgroups))}
    for i, deps in subgroup_deps.items():
        for dep in deps:
            in_degree[i] += 1

    queue = [i for i in range(len(sorted_subgroups)) if in_degree[i] == 0]
    ordered_subgroups = []

    while queue:
        queue.sort()  # Deterministic ordering
        subgroup_idx = queue.pop(0)
        ordered_subgroups.append(sorted_subgroups[subgroup_idx])

        for dependent_idx in range(len(sorted_subgroups)):
            if subgroup_idx in subgroup_deps[dependent_idx]:
                in_degree[dependent_idx] -= 1
                if in_degree[dependent_idx] == 0:
                    queue.append(dependent_idx)

    # Flatten the ordered sub-groups
    return sum(ordered_subgroups, [])


def _create_fusion_subgraph(
    graph: fx.Graph,
    region: "FusionRegion",
) -> fx.GraphModule:
    """
    Create a subgraph GraphModule for a fusion region.

    Args:
        graph: The parent graph
        region: The fusion region to wrap

    Returns:
        A GraphModule containing the subgraph
    """
    owning_module = graph.owning_module
    subgraph = fx.Graph(owning_module)

    # Map from parent graph nodes to subgraph nodes
    node_map: dict[fx.Node, fx.Node] = {}

    # Create placeholders for external inputs
    for i, ext_input in enumerate(region.external_inputs):
        placeholder = subgraph.placeholder(f"input_{i}")
        if "val" in ext_input.meta:
            placeholder.meta.update(ext_input.meta)
        node_map[ext_input] = placeholder

    # Copy region nodes into subgraph
    for node in region.nodes:
        # Map args through node_map
        def map_arg(arg, nm=node_map):
            if isinstance(arg, fx.Node):
                return nm.get(arg, arg)
            elif isinstance(arg, (list, tuple)):
                return type(arg)(map_arg(a, nm) for a in arg)
            elif isinstance(arg, dict):
                return {k: map_arg(v, nm) for k, v in arg.items()}
            return arg

        new_args = tuple(map_arg(a) for a in node.args)
        new_kwargs = {k: map_arg(v) for k, v in node.kwargs.items()}

        new_node = subgraph.call_function(
            node.target,
            new_args,
            new_kwargs,
        )
        new_node.meta.update(node.meta)
        node_map[node] = new_node

    # Output the last node (or multiple outputs if needed)
    nodes_list = list(region.nodes)
    output_node = node_map[nodes_list[-1]]
    subgraph.output(output_node)

    return fx.GraphModule(owning_module, subgraph)


def collapse_fusion_regions(
    graph: fx.Graph,
    region_of: dict[fx.Node, "FusionRegion"],
) -> tuple[dict[fx.Node, "FusionRegion"], dict[fx.Node, fx.Node]]:
    """
    Collapse fusion regions into subgraph HOP nodes.

    Each fusion region is replaced with a single call_function node that
    invokes a subgraph. The original nodes are stored in the region
    for later inlining.

    Args:
        graph: The FX graph to modify
        region_of: Mapping of nodes to their fusion regions

    Returns:
        (new_region_of, replaced) where:
        - new_region_of: Mapping from subgraph nodes to their regions
        - replaced: Mapping from original nodes to subgraph node
    """
    replaced: dict[fx.Node, fx.Node] = {}

    if not region_of:
        return region_of, replaced

    # Get unique regions
    unique_regions: list[FusionRegion] = []
    seen_region_ids: set[int] = set()
    for region in region_of.values():
        region_id = id(region)
        if region_id not in seen_region_ids:
            seen_region_ids.add(region_id)
            unique_regions.append(region)

    new_region_of: dict[fx.Node, FusionRegion] = {}
    owning_module = graph.owning_module

    for region_idx, region in enumerate(unique_regions):
        nodes_list = list(region.nodes)
        if len(nodes_list) < 2:
            # Single node region - keep as is
            if nodes_list:
                new_region_of[nodes_list[0]] = region
            continue

        last_node = nodes_list[-1]

        # Create subgraph GraphModule
        subgraph_module = _create_fusion_subgraph(graph, region)

        # Register subgraph on owning module
        subgraph_name = f"_fusion_region_{region_idx}"
        setattr(owning_module, subgraph_name, subgraph_module)

        # Create get_attr node for the subgraph
        with graph.inserting_after(last_node):
            get_subgraph = graph.get_attr(subgraph_name)

            # Create invoke_subgraph node
            subgraph_node = graph.call_function(
                torch.ops.higher_order.invoke_subgraph,
                args=(get_subgraph, subgraph_name, *tuple(region.external_inputs)),
            )
            # Copy metadata from the last node
            subgraph_node.meta.update(last_node.meta)

        # Replace all external uses of region nodes with subgraph node
        for node in nodes_list:
            for user in list(node.users.keys()):
                if user not in region.nodes and user not in (subgraph_node, get_subgraph):
                    user.replace_input_with(node, subgraph_node)
            replaced[node] = subgraph_node

        # Erase all region nodes (reverse order)
        for node in reversed(nodes_list):
            graph.erase_node(node)

        # Store subgraph info in region for later inlining
        region.subgraph_node = subgraph_node
        new_region_of[subgraph_node] = region

    return new_region_of, replaced


def expand_fusion_regions(
    graph: fx.Graph,
    region_of: dict[fx.Node, "FusionRegion"],
    replaced: dict[fx.Node, fx.Node],
) -> dict[fx.Node, fx.Node]:
    """
    Expand invoke_subgraph HOP nodes back to their original nodes.

    Args:
        graph: The FX graph
        region_of: Mapping from subgraph nodes to their fusion regions
        replaced: Mapping from original nodes to subgraph nodes (will be updated)

    Returns:
        Updated replaced mapping (original_node -> new_node)
    """
    if not region_of:
        return replaced

    owning_module = graph.owning_module

    for subgraph_node, region in region_of.items():
        if subgraph_node not in graph.nodes:
            continue

        nodes_list = list(region.nodes)
        if len(nodes_list) < 2:
            continue

        # Get the subgraph from the invoke_subgraph args
        # args = (get_attr_node, subgraph_name, *inputs)
        if len(subgraph_node.args) < 2:
            continue

        get_attr_node = subgraph_node.args[0]
        subgraph_name = subgraph_node.args[1]
        subgraph_inputs = subgraph_node.args[2:]

        # Get the subgraph module
        subgraph_module = getattr(owning_module, subgraph_name, None)
        if subgraph_module is None:
            continue

        # Map from subgraph nodes to main graph nodes
        node_map: dict[fx.Node, fx.Node] = {}

        # Map subgraph placeholders to actual inputs
        subgraph_graph = subgraph_module.graph
        placeholder_idx = 0
        for sg_node in subgraph_graph.nodes:
            if sg_node.op == "placeholder":
                if placeholder_idx < len(subgraph_inputs):
                    node_map[sg_node] = subgraph_inputs[placeholder_idx]
                placeholder_idx += 1

        # Map old region nodes to external inputs for replaced tracking
        for i, ext_input in enumerate(region.external_inputs):
            if i < len(subgraph_inputs):
                # The external input is what's passed to the subgraph
                pass  # external inputs don't need mapping in replaced

        # Inline subgraph nodes into main graph
        last_inlined_node = None
        with graph.inserting_before(subgraph_node):
            for sg_node in subgraph_graph.nodes:
                if sg_node.op == "placeholder":
                    continue
                if sg_node.op == "output":
                    continue

                # Map args through node_map
                def map_arg(arg, nm=node_map):
                    if isinstance(arg, fx.Node):
                        return nm.get(arg, arg)
                    elif isinstance(arg, (list, tuple)):
                        return type(arg)(map_arg(a, nm) for a in arg)
                    elif isinstance(arg, dict):
                        return {k: map_arg(v, nm) for k, v in arg.items()}
                    return arg

                new_args = tuple(map_arg(a) for a in sg_node.args)
                new_kwargs = {k: map_arg(v) for k, v in sg_node.kwargs.items()}

                new_node = graph.create_node(
                    op=sg_node.op,
                    target=sg_node.target,
                    args=new_args,
                    kwargs=new_kwargs,
                )
                new_node.meta.update(sg_node.meta)
                node_map[sg_node] = new_node
                last_inlined_node = new_node

        # Update replaced mapping: map original region nodes to new inlined nodes
        # We need to match by position in the region
        sg_call_nodes = [n for n in subgraph_graph.nodes if n.op == "call_function"]
        for i, old_node in enumerate(nodes_list):
            if i < len(sg_call_nodes):
                sg_node = sg_call_nodes[i]
                if sg_node in node_map:
                    replaced[old_node] = node_map[sg_node]

        # Replace uses of subgraph node with the last inlined node
        if last_inlined_node is not None:
            subgraph_node.replace_all_uses_with(last_inlined_node)

        # Erase the subgraph node and get_attr
        graph.erase_node(subgraph_node)
        if get_attr_node.users == {}:
            graph.erase_node(get_attr_node)

        # Clean up the subgraph attribute
        if hasattr(owning_module, subgraph_name):
            delattr(owning_module, subgraph_name)

    return replaced


def resolve_replacement_chain(
    node: fx.Node,
    replaced: dict[fx.Node, fx.Node],
) -> fx.Node:
    """Follow replacement chain to get the final node."""
    visited = set()
    while node in replaced and node not in visited:
        visited.add(node)
        node = replaced[node]
    return node