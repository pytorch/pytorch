"""Detect fusion regions for overlap scheduling."""

from dataclasses import dataclass, field

import torch
import torch.fx as fx
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.utils._ordered_set import OrderedSet
from torch.utils._runtime_estimation import get_num_bytes


@dataclass
class FusionRegion:
    """Represents a connected set of fusible operations that will fuse together."""

    nodes: OrderedSet[fx.Node]  # All nodes in topo order
    cost_ms: float = field(default=0.0, init=False)  # Estimated cost in milliseconds
    subgraph_node: fx.Node | None = field(default=None, init=False)

    def compute_cost(self, subgraph: fx.GraphModule) -> None:
        """Compute cost based on subgraph's placeholder inputs and output node."""
        from torch._inductor.utils import get_gpu_dram_gbps

        def get_node_bytes(node: fx.Node) -> int:
            val = node.meta.get("val")
            return get_num_bytes(val) if isinstance(val, torch.Tensor) else 0

        total_bytes = 0

        for node in subgraph.graph.nodes:
            if node.op == "placeholder":
                total_bytes += get_node_bytes(node)
            elif node.op == "output":
                for arg in node.args[0] if isinstance(node.args[0], (list, tuple)) else [node.args[0]]:
                    if isinstance(arg, fx.Node):
                        total_bytes += get_node_bytes(arg)

        if total_bytes > 0:
            fusion_bw_gbps = get_gpu_dram_gbps()
            fusion_bw_bytes_per_s = fusion_bw_gbps * 1e9
            self.cost_ms = (total_bytes / fusion_bw_bytes_per_s) * 1000


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
        # partition.nodes is already sorted topologically by the partitioner
        nodes_list = list(partition.nodes.keys())

        # Skip regions with fewer than 2 non-view nodes (views have no cost)
        non_view_count = sum(1 for n in nodes_list if not is_view_node(n))
        if non_view_count < 2:
            continue

        region = FusionRegion(nodes=OrderedSet(nodes_list))

        for node in nodes_list:
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

    # Get unique regions (regions with <2 nodes already filtered in build_fusion_regions)
    unique_regions: list[FusionRegion] = []
    seen_region_ids: set[int] = set()
    for region in region_of.values():
        region_id = id(region)
        if region_id not in seen_region_ids:
            seen_region_ids.add(region_id)
            unique_regions.append(region)

    if not unique_regions:
        return region_of, replaced

    # Build partitions list for fuse_by_partitions
    partitions = [{node: None for node in region.nodes} for region in unique_regions]

    # Fuse all partitions at once
    fuse_by_partitions(gm, partitions, prefix="_fusion_region_")

    # Build replaced mapping and new_region_of by finding the call_module nodes
    new_region_of: dict[fx.Node, FusionRegion] = {}

    for region_idx, region in enumerate(unique_regions):
        subgraph_name = f"_fusion_region_{region_idx}"

        # Find the call_module node
        module_node = None
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target == subgraph_name:
                module_node = node
                break

        if module_node is None:
            continue

        # Compute cost from the subgraph's placeholders and outputs
        subgraph_module = getattr(gm, subgraph_name)
        region.compute_cost(subgraph_module)

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
    fusion_replaced: dict[fx.Node, fx.Node],
) -> dict[fx.Node, fx.Node]:
    """
    Expand call_module nodes back to their original nodes using _inline_module.
    Returns mapping from module_node -> output_node (only the output, not all nodes).
    """
    from torch.fx.experimental.const_fold import _inline_module

    if not region_of:
        return {}

    # Only map module nodes to their output nodes (not all intermediate nodes)
    module_to_output: dict[fx.Node, fx.Node] = {}

    for module_node, region in list(region_of.items()):
        if module_node.op != "call_module":
            continue

        subgraph_name = module_node.target
        if not hasattr(gm, subgraph_name):
            continue

        # _inline_module returns subgraph_node -> new_node mapping
        subgraph_to_new = _inline_module(gm, subgraph_name)

        # Find the last call_function node (the output of the fusion region)
        last_new_call = None
        for n, new_n in subgraph_to_new.items():
            if new_n.op == "call_function":
                last_new_call = new_n

        # Only map the module node to its output
        if last_new_call:
            module_to_output[module_node] = last_new_call

        # Remove the submodule attribute (if _inline_module didn't remove it)
        if hasattr(gm, subgraph_name):
            delattr(gm, subgraph_name)

    return module_to_output
