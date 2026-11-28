"""Detect fusion regions for overlap scheduling."""

from math import prod
from typing import cast

import torch
import torch.fx as fx
from torch._inductor import config
from torch._inductor.fx_passes.overlap_scheduling import (
    estimate_roofline_runtime_ms,
    is_compute_node,
)
from torch._inductor.fx_utils import get_mutated_input_nodes, get_storage
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten
from torch.utils._runtime_estimation import get_transfer_time


def is_uninitialized_allocation(n: fx.Node) -> bool:
    if n.op != "call_function" or n.target not in (
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten.new_empty.default,
        torch.ops.aten.new_empty_strided.default,
    ):
        return False
    value = n.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and type(value) in (torch.Tensor, FakeTensor)
        and value.layout == torch.strided
        and not value.is_nested
        and value.ndim > 0
        and not (
            torch.are_deterministic_algorithms_enabled()
            and torch._C._get_deterministic_fill_uninitialized_memory()
        )
    )


def is_view_node(n: fx.Node) -> bool:
    """Check if a node is a view operation (zero cost, no memory allocation)."""

    return isinstance(n.target, torch._ops.OpOverload) and (
        n.target.is_view and n.target.namespace in ("aten", "prims")
    )


def _has_fusible_lowering(n: fx.Node) -> bool:
    if n.op != "call_function":
        return False

    target = n.target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    # Exclude collectives and waits (they have their own scheduling)
    if target.namespace == "_c10d_functional":
        return False

    from torch._inductor.lowering import fallbacks, lowerings

    # Must have a lowering
    if target not in lowerings:
        return False

    # Exclude fallbacks (ops that fall back to eager execution)
    if target in fallbacks:
        return False

    # Special case: cat is only fusible if it has few enough inputs
    if target == torch.ops.aten.cat.default:
        inputs = n.args[0] if n.args else []
        if isinstance(inputs, (list, tuple)):
            if len(inputs) > config.max_pointwise_cat_inputs:
                return False

    return True


def is_fusible_node(n: fx.Node) -> bool:
    """Check if a node is fusible based on whether it has an inductor lowering.

    A node is fusible if:
    - It has a lowering in torch._inductor.lowering.lowerings
    - It does NOT have a flop counter (expensive compute ops like mm/conv)
    - It is NOT a registered fallback (ops that fall back to eager)
    - It is NOT a collective or wait op
    - For aten.cat, it must have <= max_pointwise_cat_inputs inputs

    Supported GEMMs are an exception to the flop counter rule: they can fuse
    producers, epilogues, or native matmul operations when enabled.
    Final fusion decisions remain with the lowered Inductor scheduler.
    """
    if not _has_fusible_lowering(n):
        return False
    target = cast(torch._ops.OpOverload, n.target)
    if torch.Tag.nondeterministic_seeded in target.tags:
        return False
    custom = n.meta.get("custom")
    if n.meta.get("should_fallback") or (
        isinstance(custom, dict) and custom.get("fallback_to_eager")
    ):
        return False
    if is_compute_node(n):
        return target.overloadpacket in (
            torch.ops.aten.mm,
            torch.ops.aten.bmm,
            torch.ops.aten.addmm,
            torch.ops.aten.baddbmm,
            torch.ops.aten._scaled_mm,
        ) and (
            config.prologue_fusion
            or config.epilogue_fusion
            or config.triton.native_matmul
        )
    return True


def _has_fusion_cost(n: fx.Node, prioritize: bool = True) -> bool:
    return (
        n.op == "call_function"
        and not is_view_node(n)
        and (not prioritize or not is_uninitialized_allocation(n))
    )


def _horizontal_domain(n: fx.Node) -> tuple[object, ...] | None:
    if not _has_fusion_cost(n) or not isinstance(n.target, torch._ops.OpOverload):
        return None
    val = n.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None
    shape = tuple(optimization_hint(size, fallback=-1) for size in val.shape)
    if -1 in shape:
        return None
    if is_compute_node(n):
        if not config.triton.native_matmul:
            return None
        tensors = [inp.meta.get("val") for inp in n.all_input_nodes]
        # addmm/baddbmm include a bias input before the matrix operands.
        offset = int(
            n.target.overloadpacket in (torch.ops.aten.addmm, torch.ops.aten.baddbmm)
        )
        matrix = tensors[offset] if len(tensors) > offset else None
        if not isinstance(matrix, torch.Tensor) or matrix.ndim < 2:
            return None
        k = optimization_hint(matrix.shape[-1], fallback=-1)
        return (val.device, shape, "matmul", k) if k >= 0 else None
    if torch.Tag.reduction in n.target.tags:
        tensors = [inp.meta.get("val") for inp in n.all_input_nodes]
        value = tensors[0] if tensors else None
        if not isinstance(value, torch.Tensor):
            return None
        input_shape = tuple(
            optimization_hint(size, fallback=-1) for size in value.shape
        )
        if -1 in input_shape:
            return None
        return val.device, prod(shape), "reduction", prod(input_shape)
    return val.device, prod(shape), "pointwise"


def _can_attach_matmul(producer: fx.Node, consumer: fx.Node) -> bool:
    a, b = producer.meta.get("val"), consumer.meta.get("val")
    if (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.device != b.device
    ):
        return False
    if is_compute_node(producer) and is_compute_node(consumer):
        return False
    if is_compute_node(consumer):
        # Accept prologues without mutations or reductions.
        target = producer.target
        return not isinstance(target, torch._ops.OpOverload) or (
            (config.triton.native_matmul or config.prologue_fusion)
            and not target._schema.is_mutable
            and torch.Tag.reduction not in target.tags
        )
    if is_compute_node(producer):
        # Accept views and epilogues; reductions require native matmul.
        target = consumer.target
        return is_view_node(consumer) or (
            isinstance(target, torch._ops.OpOverload)
            and (config.triton.native_matmul or config.epilogue_fusion)
            and (torch.Tag.reduction not in target.tags or config.triton.native_matmul)
        )
    return True


def _get_contiguous_fusible_spans(
    gm: fx.GraphModule, *, prioritize: bool = False
) -> list[list[fx.Node]]:
    """Get contiguous spans of fusible nodes from the graph.

    Walks the graph in topological order and groups consecutive fusible
    nodes into spans. Non-fusible nodes act as span boundaries.
    """
    from torch.utils.flop_counter import flop_registry

    spans: list[list[fx.Node]] = []
    current_span: list[fx.Node] = []

    for node in gm.graph.nodes:
        fusible = is_fusible_node(node) if prioritize else _has_fusible_lowering(node)
        if fusible and not prioritize:
            target = cast(torch._ops.OpOverload, node.target)
            # Preserve the original exclusion of expensive compute operations.
            fusible = target.overloadpacket not in flop_registry
        if fusible:
            current_span.append(node)
        else:
            # Non-fusible node ends the current span
            if current_span:
                spans.append(current_span)
                current_span = []

    if current_span:
        spans.append(current_span)

    return spans


def _find_connected_components(
    span: list[fx.Node], *, prioritize: bool = False
) -> list[list[fx.Node]]:
    """Find connected components within a span of fusible nodes.

    Two nodes are connected if one is an input to the other (direct data dependency).
    With fusion prioritization, also join compatible horizontal candidates.
    """
    if not span:
        return []

    from torch.fx.experimental.optimization import UnionFind

    span_set = OrderedSet(span)
    node_to_idx = {n: i for i, n in enumerate(span)}

    uf = UnionFind(len(span))
    for i in range(len(span)):
        uf.make_set(i)

    if prioritize:
        sizes = [int(_has_fusion_cost(n)) for n in span]
        matmuls = [int(is_compute_node(n)) for n in span]
        matmul_domains = [
            _horizontal_domain(n) if is_compute_node(n) else None for n in span
        ]
        domains: dict[tuple[object, ...], int] = {}
        native_matmul = config.triton.native_matmul

        def join(
            left: int,
            right: int,
            *,
            producer_edge: bool = False,
            horizontal: bool = False,
        ) -> None:
            a, b = uf.find(left), uf.find(right)
            if a == b or sizes[a] + sizes[b] > config.max_fusion_size:
                return
            if horizontal and (matmuls[a] or matmuls[b]) and not native_matmul:
                return
            if matmuls[a] and matmuls[b]:
                if (
                    producer_edge
                    or not native_matmul
                    or matmul_domains[a] is None
                    or matmul_domains[a] != matmul_domains[b]
                ):
                    return
            count, compute = sizes[a] + sizes[b], matmuls[a] + matmuls[b]
            domain = matmul_domains[a] if matmuls[a] else matmul_domains[b]
            uf.join(a, b)
            root = uf.find(a)
            sizes[root], matmuls[root], matmul_domains[root] = count, compute, domain

        for node in span:
            node_idx = node_to_idx[node]
            for inp in node.all_input_nodes:
                if inp in span_set and _can_attach_matmul(inp, node):
                    join(
                        node_to_idx[inp], node_idx, producer_edge=is_compute_node(node)
                    )
            domain = _horizontal_domain(node)
            if domain is not None:
                if domain in domains:
                    join(domains[domain], node_idx, horizontal=True)
                domains[domain] = node_idx

    else:
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
    *,
    prioritize: bool = False,
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """Build fusion regions from contiguous spans of fusible nodes.

    1. Identify contiguous spans of fusible nodes (separated by non-fusible nodes)
    2. Find connected components within each span
    3. Return regions that have 2+ non-view nodes

    With fusion prioritization, also join compatible horizontal operations and
    omit uninitialized allocations from the count of nodes with fusion cost.

    This ensures fusion regions are strictly local - no reordering across
    non-fusible node boundaries.

    Returns a dict mapping each node to its fusion group (OrderedSet of nodes).
    """
    # Build node -> topo index map for sorting
    node_to_idx: dict[fx.Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # Step 1: Get contiguous spans of fusible nodes
    spans = _get_contiguous_fusible_spans(gm, prioritize=prioritize)

    # Step 2: Find connected components within each span
    region_of: dict[fx.Node, OrderedSet[fx.Node]] = {}

    for span in spans:
        if len(span) < 2:
            continue

        components = _find_connected_components(span, prioritize=prioritize)

        for component in components:
            # Skip regions with fewer than 2 non-view nodes (views have no cost)
            if sum(_has_fusion_cost(n, prioritize) for n in component) < 2:
                continue

            # Sort nodes in topological order to preserve original ordering
            sorted_component = sorted(component, key=lambda n: node_to_idx[n])
            node_set = OrderedSet(sorted_component)

            for node in sorted_component:
                region_of[node] = node_set

    return region_of


def contiguous_fusion_regions(gm: fx.GraphModule) -> list[OrderedSet[fx.Node]]:
    region_of = build_fusion_regions(gm, prioritize=True)
    regions = []
    current: OrderedSet[fx.Node] = OrderedSet()
    previous = None
    for node in gm.graph.nodes:
        region = region_of.get(node)
        if region is not previous:
            if len(current) > 1:
                regions.append(current)
            current = OrderedSet()
        if region is not None:
            current.add(node)
        previous = region
    if len(current) > 1:
        regions.append(current)
    return [
        region for region in regions if sum(_has_fusion_cost(n) for n in region) > 1
    ]


def _collect_fusion_boundary_ins_outs(
    node: fx.Node, region: OrderedSet[fx.Node], *, prioritize: bool = True
) -> tuple[list[object], list[object]]:
    r"""Collect external inputs, external outputs, and mutation destinations.

    Example::
        INPUT:
            x = placeholder()               # storage 1
            y = placeholder()               # storage 2
            sine = aten.sin(y)              # storage 3
            copied = aten.copy_(x, sine)     # storage 1
            result = output(copied)

            node = copied
            region = {sine, copied}

        OUTPUT:
            inputs = [x.meta["val"]]
            outputs = [x.meta["val"]]
    """
    # External inputs: values from nodes outside this region
    ext_inputs = []
    for inp in node.all_input_nodes:
        if inp not in region:
            val = inp.meta.get("val")
            if val is not None:
                ext_inputs.append(val)

    # External outputs: this node's value is used outside the region
    has_external_user = any(u not in region for u in node.users)
    if not prioritize:
        val = node.meta.get("val")
        return ext_inputs, [val] if has_external_user and val is not None else []
    written = OrderedSet(get_mutated_input_nodes(node))
    ext_outputs = [inp.meta["val"] for inp in written if "val" in inp.meta]
    flat_written, _ = tree_flatten(ext_outputs)
    written_storages = OrderedSet(
        get_storage(t)
        for t in flat_written
        if isinstance(t, torch.Tensor) and torch._C._has_storage(t)
    )
    if has_external_user:
        val = node.meta.get("val")
        if val is not None:
            outputs, _ = tree_flatten(val)
            ext_outputs.extend(
                t
                for t in outputs
                if not isinstance(t, torch.Tensor)
                or not torch._C._has_storage(t)
                or get_storage(t) not in written_storages
            )
    return ext_inputs, ext_outputs


def estimate_fused_node_costs(
    region_of: dict[fx.Node, OrderedSet[fx.Node]],
    *,
    prioritize: bool = False,
) -> dict[fx.Node, float]:
    """Estimate per-node costs accounting for fusion (no graph collapse).

    For each node in a fusion region, only count I/O that crosses the
    region boundary (external reads/writes). Internal intermediates
    won't be materialized by inductor, so they shouldn't count.

    With fusion prioritization, also retain GEMM runtime and destination writes.

    Nodes with only internal I/O get cost 0. Nodes with external I/O
    get the bandwidth cost of just those external tensors.
    """
    costs: dict[fx.Node, float] = {}
    seen: OrderedSet[int] = OrderedSet()

    for node_set in region_of.values():
        rid = id(node_set)
        if rid in seen:
            continue
        seen.add(rid)

        for n in node_set:
            if prioritize and is_compute_node(n):
                costs[n] = estimate_roofline_runtime_ms(n)
                continue
            if prioritize and (n.op == "get_attr" or is_uninitialized_allocation(n)):
                costs[n] = 0.0
                continue
            ext_inputs, ext_outputs = _collect_fusion_boundary_ins_outs(
                n, node_set, prioritize=prioritize
            )

            if not ext_inputs and not ext_outputs:
                costs[n] = 0.0
            else:
                flat_in, _ = tree_flatten(ext_inputs)
                flat_out, _ = tree_flatten(ext_outputs)
                costs[n] = get_transfer_time(flat_in, flat_out) / 1e6

    return costs
