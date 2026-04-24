"""AC rematerialize pass: Duplicates recompute nodes for backward, then DCE removes unused forward versions."""

import logging

import torch
import torch.fx as fx
from torch._functorch.compile_utils import raise_getitems
from torch._functorch.partitioners import (
    cleanup_recompute_tags,
    force_save_bw_mutation_src,
    has_recomputable_ops,
    has_recomputable_rng_ops,
    is_not_collective,
    must_recompute,
)


log = logging.getLogger(__name__)
_EMPTY_CUSTOM_META: dict[str, object] = {}


def is_impure_node_for_dce(node: fx.Node) -> bool:
    # Check for special collectives that should be treated as pure
    if not is_not_collective(node):
        # It's a collective (wait_tensor, all_gather_into_tensor, etc.)
        # Treat as pure - can be eliminated if unused
        return False

    # For everything else, fall back to the DEFAULT logic
    # This is what eliminate_dead_code() calls when is_impure_node=None
    impure_random = True
    if torch._guards.TracingContext.try_get():
        impure_random = torch._inductor.config.fallback_random
    return node.is_impure(impure_random)


def _is_backward_node(node: fx.Node, use_phase: bool = False) -> bool:
    """Check if node is in backward region.

    If use_phase is True, only checks custom["phase"] == "backward"
    (user annotation). Otherwise falls back to node.meta["autograd_backward"],
    which Dynamo adds when tracing torch.autograd.grad.
    """
    custom = node.meta.get("custom", _EMPTY_CUSTOM_META)
    if use_phase:
        return custom.get("phase") == "backward"
    return node.meta.get("autograd_backward", False)


def _has_user_phase_annotation(gm: fx.GraphModule) -> bool:
    """Check if any node has the user-level phase: backward annotation."""
    return any(
        node.meta.get("custom", _EMPTY_CUSTOM_META).get("phase") == "backward"
        for node in gm.graph.nodes
    )


def _collect_backward_regions(
    gm: fx.GraphModule, use_phase: bool
) -> list[tuple[int, int, bool]]:
    """Returns (bwd_start, bwd_end, needs_remat) for each backward region.

    Regions are maximal contiguous runs of backward nodes, as [start, end)
    indices into the graph node list.
    """
    regions: list[tuple[int, int, bool]] = []
    bwd_start: int | None = None
    needs_remat = False

    for idx, node in enumerate(gm.graph.nodes):
        if _is_backward_node(node, use_phase=use_phase):
            if bwd_start is None:
                bwd_start = idx
                needs_remat = False
            if not needs_remat and any(
                must_recompute(inp) for inp in node.all_input_nodes
            ):
                needs_remat = True
        elif bwd_start is not None:
            regions.append((bwd_start, idx, needs_remat))
            bwd_start = None

    if bwd_start is not None:
        regions.append((bwd_start, idx + 1, needs_remat))

    return regions


def remat_using_tags_for_fwd_loss_bwd_graph(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    """
    Duplicate recompute nodes for backward use. DCE removes unused forward versions.

    Backward regions are identified by custom["phase"] == "backward" (user
    annotation) or node.meta["autograd_backward"] == True (set automatically when
    Dynamo traces torch.autograd.grad). When the user provides phase
    annotations, only those annotated regions are used.

    The graph may contain multiple disjoint backward regions (e.g. chunked
    loss). Regions that do not depend on recomputable forward nodes are
    skipped. Only one region may require remat; if multiple do, we error
    and ask the user to annotate which region to rematerialize.
    """
    if not has_recomputable_ops(gm):
        return gm

    if has_recomputable_rng_ops(gm):
        raise RuntimeError(
            "Activation checkpoint rematerialization in `forward-loss-backward` graph does not support RNG ops "
            "in recompute regions. Please move RNG operations outside "
            "of recompute regions, or use joint graph mode (where partitioner handles RNG)."
        )

    # Use partitioner pass to normalize AC node tags.
    gm = cleanup_recompute_tags(gm, is_default_partition=True)

    force_save_bw_mutation_src(gm)

    # must_recompute (used inside _collect_backward_regions) requires
    # cleanup_recompute_tags to have run first.
    use_phase = _has_user_phase_annotation(gm)
    regions = _collect_backward_regions(gm, use_phase)
    if not regions:
        return gm

    # User-annotated phase regions: multiple annotations is always an error.
    if use_phase and len(regions) > 1:
        raise RuntimeError(
            f"Detected {len(regions)} disjoint backward regions annotated with "
            'phase: "backward" but remat only supports a single backward region. '
            "Please ensure only one contiguous region is annotated."
        )

    remat_regions = [(s, e) for s, e, needs in regions if needs]

    if len(remat_regions) > 1:
        raise RuntimeError(
            f"Detected {len(remat_regions)} disjoint backward regions that require recomputation, "
            "but remat only supports one such region in a forward-loss-backward graph."
        )

    if not remat_regions:
        return gm

    bwd_start, bwd_end = remat_regions[0]

    nodes_list = list(gm.graph.nodes)
    order = {node: idx for idx, node in enumerate(nodes_list)}
    recomputed_nodes: dict[fx.Node, fx.Node] = {}

    def remat_input(x: fx.Node) -> fx.Node:
        return recomputed_nodes.get(x, x)

    def gather_recompute_deps(node: fx.Node) -> set[fx.Node]:
        deps: set[fx.Node] = set()

        def _gather(n: fx.Node) -> None:
            if n in deps or n in recomputed_nodes or not must_recompute(n):
                return
            deps.add(n)
            for inp in n.all_input_nodes:
                _gather(inp)

        # Can't call _gather(node) directly: node itself may not be must_recompute
        # (e.g. backward nodes), so _gather would return early without visiting inputs.
        for inp in node.all_input_nodes:
            _gather(inp)
        return deps

    for node in nodes_list[bwd_start:bwd_end]:
        deps = gather_recompute_deps(node)

        new_deps = sorted(deps, key=lambda n: order[n])
        if new_deps:
            log.debug(
                "To compute backward node %s, recomputing [%s]",
                node.name,
                ", ".join(dep.name for dep in new_deps),
            )
        for dep in new_deps:
            with gm.graph.inserting_before(node):
                dup = gm.graph.node_copy(dep, remat_input)
                dup.name = dep.name + "_recomputed"
                dup.meta["autograd_backward"] = True
                recomputed_nodes[dep] = dup

        for inp in list(node.all_input_nodes):
            if inp in recomputed_nodes:
                node.replace_input_with(inp, recomputed_nodes[inp])

    # DCE with custom is_impure_node (like default_partition)
    # Treats certain collectives as pure while delegating to default impurity logic
    gm.graph.eliminate_dead_code(is_impure_node=is_impure_node_for_dce)

    # raise_getitems pass for better memory (like default_partition)
    gm = raise_getitems(gm)

    gm.recompile()

    return gm
