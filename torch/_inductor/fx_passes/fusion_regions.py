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


def inline_subgraphs(
    graph: fx.Graph,
    region_of: dict[fx.Node, any],
    dep_map: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """
    Inline subgraph nodes back into the main graph and transfer dependencies.

    If subgraph nodes were created for fusion regions, this function:
    1. Re-inserts the original region nodes back into the graph
    2. Maps dependencies to/from subgraph nodes to the last node in the region
    3. Preserves metadata during the inlining process

    Args:
        graph: The FX graph
        region_of: Mapping of nodes to their fusion regions (with subgraph_node field)
        dep_map: Dependencies between nodes

    Returns:
        Updated dep_map with subgraph nodes replaced by their last internal node
    """
    # Early exit if no regions
    if not region_of:
        return dep_map

    # Get unique regions that have subgraph nodes
    regions_with_subgraphs = []
    seen_region_ids = OrderedSet()

    for region in region_of.values():
        region_id = id(region)
        if (
            region_id not in seen_region_ids
            and hasattr(region, "subgraph_node")
            and region.subgraph_node is not None
        ):
            seen_region_ids.add(region_id)
            regions_with_subgraphs.append(region)

    # Early exit if no subgraphs to inline
    if not regions_with_subgraphs:
        return dep_map

    # Inline each subgraph
    for region in regions_with_subgraphs:
        subgraph_node = region.subgraph_node
        if subgraph_node not in graph.nodes:
            continue

        # Re-insert all nodes from the region back into the graph
        with graph.inserting_before(subgraph_node):
            node_map: dict[fx.Node, fx.Node] = {}

            # Map external inputs (they're already in the graph)
            for ext_input in region.external_inputs:
                node_map[ext_input] = ext_input

            # Clone each node in the region back into the graph
            for old_node in region.nodes:
                # Create the new node using node_copy
                new_node = graph.node_copy(old_node, lambda n: node_map.get(n, n))
                node_map[old_node] = new_node
                # Transfer metadata (following control_dependencies.py pattern)
                new_node.meta.update(old_node.meta)

        # Get the last node (output of the region)
        last_node = node_map[list(region.nodes)[-1]]

        # Replace uses of subgraph_node with the appropriate output
        if len(region.external_outputs) == 1:
            output_node = node_map[next(iter(region.external_outputs))]
            subgraph_node.replace_all_uses_with(output_node)
        else:
            # Multiple outputs - find getitem nodes and replace them
            for user in list(subgraph_node.users.keys()):
                if user.target == torch.ops.aten.getitem:
                    idx = user.args[1]
                    output_node = node_map[list(region.external_outputs)[idx]]
                    user.replace_all_uses_with(output_node)
                    graph.erase_node(user)

        # Erase the subgraph node
        graph.erase_node(subgraph_node)

        # Update dep_map: replace subgraph_node references with last_node
        new_dep_map: dict[fx.Node, OrderedSet[fx.Node]] = {}
        for node, deps in dep_map.items():
            new_node = last_node if node == subgraph_node else node
            new_deps = OrderedSet()
            for dep in deps:
                new_dep = last_node if dep == subgraph_node else dep
                new_deps.add(new_dep)

            if new_node in new_dep_map:
                new_dep_map[new_node].update(new_deps)
            else:
                new_dep_map[new_node] = new_deps

        dep_map = new_dep_map

    return dep_map