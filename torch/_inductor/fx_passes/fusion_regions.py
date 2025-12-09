"""Detect fusion regions for overlap scheduling."""

from dataclasses import dataclass, field

import torch
import torch.fx as fx
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.utils._ordered_set import OrderedSet


@dataclass
class FusionRegion:
    """Represents a connected set of fusible operations that will fuse together."""

    nodes: OrderedSet[fx.Node]  # All nodes in topo order
    cost_ms: float = field(default=0.0, init=False)  # Estimated cost in milliseconds
    subgraph_node: fx.Node | None = field(default=None, init=False)

    def __post_init__(self):
        """Compute cost based on external I/O (bandwidth-bound estimate)."""
        from torch._inductor.utils import get_gpu_dram_gbps

        region_set = set(self.nodes)

        def get_tensor_bytes(node: fx.Node) -> int:
            """Get tensor bytes, accounting for expanded dims (stride=0)."""
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor) or val.numel() == 0:
                return 0
            # Account for expanded/broadcast dims by computing actual storage span
            # For dims with stride=0, only 1 element is accessed regardless of size
            real_numel = 1
            for size, stride in zip(val.shape, val.stride()):
                if stride != 0:
                    real_numel *= size
            return real_numel * val.element_size()

        # Compute total bytes of external inputs and outputs
        total_bytes = 0
        seen_nodes: set[fx.Node] = set()

        for node in self.nodes:
            # External inputs: inputs from outside the region
            for inp in node.all_input_nodes:
                if inp not in region_set and inp not in seen_nodes:
                    seen_nodes.add(inp)
                    total_bytes += get_tensor_bytes(inp)

            # External outputs: nodes with users outside the region
            if node not in seen_nodes and (
                any(u not in region_set for u in node.users) or len(node.users) == 0
            ):
                seen_nodes.add(node)
                total_bytes += get_tensor_bytes(node)

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


def is_view_node(n: fx.Node) -> bool:
    """Check if a node is a view operation (no memory read/write)."""
    if n.op != "call_function":
        return False
    return getattr(n.target, "is_view", False)


def is_fusible_node(n: fx.Node) -> bool:
    """Check if a node is fusible (pointwise, reduction, views, indexing ops, small cats).

    Excludes: mm/conv, collectives, waits, placeholders, outputs.
    """
    if n.op != "call_function":
        return False

    # Include pointwise, reduction, views
    tags = getattr(n.target, "tags", ())
    if torch.Tag.pointwise in tags or torch.Tag.reduction in tags:
        return True

    if is_view_node(n):
        return True

    aten = torch.ops.aten

    # Include cat if inputs within pointwise cat threshold
    if n.target == aten.cat.default:
        inputs = n.args[0] if n.args else []
        if isinstance(inputs, (list, tuple)):
            import torch._inductor.config as inductor_config

            if len(inputs) <= inductor_config.max_pointwise_cat_inputs:
                return True

    # Include constant_pad_nd (can be pointwise)
    if n.target == aten.constant_pad_nd.default:
        return True

    # Include specific indexing ops
    if n.target in (aten.slice.Tensor, aten.gather.default, aten.embedding.default):
        return True

    return False


class FusibleOperatorSupport(OperatorSupportBase):
    """Operator support for fusible operations (pointwise, reduction, view)."""

    def is_node_supported(
        self, submodules: dict[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        return is_fusible_node(node)


def build_fusion_regions(
    gm: fx.GraphModule,
) -> dict[fx.Node, FusionRegion]:
    """Build fusion regions using CapabilityBasedPartitioner with skip_horizontal_fusion.

    Returns a dict mapping each node to its containing region (if any).
    """
    from torch.fx.passes.utils.fuser_utils import topo_sort

    operator_support = FusibleOperatorSupport()
    partitioner = CapabilityBasedPartitioner(
        gm,
        operator_support,
        allows_single_node_partition=False,
        skip_horizontal_fusion=True,
    )

    partitions = partitioner.propose_partitions()

    # Convert partitions to FusionRegion objects
    region_of: dict[fx.Node, FusionRegion] = {}

    for partition in partitions:
        nodes_list = list(partition.nodes.keys())
        if len(nodes_list) < 2:
            continue

        # Sort nodes topologically
        nodes_sorted = topo_sort(nodes_list)
        region = FusionRegion(nodes=OrderedSet(nodes_sorted))

        # Only include regions with positive cost (have tensor I/O)
        if region.cost_ms > 0:
            for node in nodes_sorted:
                region_of[node] = region

    return region_of


def collapse_fusion_regions(
    gm: fx.GraphModule,
    region_of: dict[fx.Node, FusionRegion],
) -> tuple[dict[fx.Node, FusionRegion], dict[fx.Node, fx.Node]]:
    """
    Collapse fusion regions into call_module nodes using fuse_by_partitions.
    Returns (new_region_of, replaced) mapping module nodes to regions and
    original nodes to their replacement module node.
    """
    from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

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

    # Filter to regions with 2+ nodes
    valid_regions = [r for r in unique_regions if len(r.nodes) >= 2]

    if not valid_regions:
        return region_of, replaced

    # Build partitions list for fuse_by_partitions
    partitions = [{node: None for node in region.nodes} for region in valid_regions]

    # Fuse all partitions at once
    fuse_by_partitions(gm, partitions, prefix="_fusion_region_")

    # Build replaced mapping and new_region_of by finding the call_module nodes
    new_region_of: dict[fx.Node, FusionRegion] = {}

    for region_idx, region in enumerate(valid_regions):
        subgraph_name = f"_fusion_region_{region_idx}"

        # Find the call_module node
        module_node = None
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target == subgraph_name:
                module_node = node
                break

        if module_node is None:
            continue

        # Map original nodes to module node
        for node in region.nodes:
            replaced[node] = module_node

        # Store module info in region
        region.subgraph_node = module_node
        new_region_of[module_node] = region

    return new_region_of, replaced


def expand_fusion_regions(
    gm: fx.GraphModule,
    region_of: dict[fx.Node, FusionRegion],
    replaced: dict[fx.Node, fx.Node],
) -> dict[fx.Node, fx.Node]:
    """
    Expand call_module nodes back to their original nodes using _inline_module.
    Returns updated replaced mapping.
    """
    from torch.fx.experimental.const_fold import _inline_module

    if not region_of:
        return replaced

    for module_node, region in list(region_of.items()):
        if module_node.op != "call_module":
            continue

        subgraph_name = module_node.target
        subgraph_module = getattr(gm, subgraph_name, None)
        if subgraph_module is None:
            continue

        # Track nodes before inlining to identify new nodes after
        nodes_before = set(gm.graph.nodes)

        # Get subgraph call_function nodes in order (to match with inlined nodes)
        subgraph_call_nodes = [
            n for n in subgraph_module.graph.nodes if n.op == "call_function"
        ]

        # Use existing _inline_module utility to inline the subgraph
        _inline_module(gm, subgraph_name)

        # Find newly inlined nodes (nodes that weren't in the graph before)
        new_nodes = [n for n in gm.graph.nodes if n not in nodes_before]
        new_call_nodes = [n for n in new_nodes if n.op == "call_function"]

        # Map original region nodes to inlined nodes by position
        # fuse_as_graphmodule preserves node order, so we can match by index
        nodes_list = list(region.nodes)
        for i, orig_node in enumerate(nodes_list):
            if i < len(new_call_nodes):
                replaced[orig_node] = new_call_nodes[i]

        # Map module_node to the last inlined node (the output)
        if new_call_nodes:
            replaced[module_node] = new_call_nodes[-1]

        # Remove the submodule attribute (if _inline_module didn't remove it)
        if hasattr(gm, subgraph_name):
            delattr(gm, subgraph_name)

    return replaced
