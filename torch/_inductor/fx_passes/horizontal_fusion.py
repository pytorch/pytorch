"""Horizontal fusion: convert repeated identical ops to foreach multi-tensor kernels.

Detects groups of N independent identical ops (same target, same scalar args,
different tensor args, same shapes) and rewrites them using _foreach_* ops
which inductor lowers to multi-tensor kernels via ForeachKernelSchedulerNode.

This eliminates CPU kernel launch overhead for patterns like optimizer
per-parameter updates (Muon, Adam, LAMB).
"""

import logging
import operator
from collections import defaultdict

import torch
import torch.fx as fx

log = logging.getLogger(__name__)

aten = torch.ops.aten

# Binary ops: op(tensor, tensor_or_scalar) -> tensor
# Maps to (foreach_scalar_variant, foreach_list_variant)
_FUSEABLE_BINARY_OPS = {
    aten.mul.Tensor: (aten._foreach_mul.Scalar, aten._foreach_mul.List),
    aten.add.Tensor: (aten._foreach_add.Scalar, aten._foreach_add.List),
    aten.sub.Tensor: (aten._foreach_sub.Scalar, aten._foreach_sub.List),
    aten.div.Tensor: (aten._foreach_div.Scalar, aten._foreach_div.List),
    aten.pow.Tensor_Scalar: (aten._foreach_pow.Scalar, aten._foreach_pow.List),
    aten.clamp_min.default: (aten._foreach_clamp_min.Scalar, aten._foreach_clamp_min.List),
}

# Unary ops: op(tensor) -> tensor
_FUSEABLE_UNARY_OPS = {
    aten.sqrt.default: aten._foreach_sqrt.default,
    aten.neg.default: aten._foreach_neg.default,
}

# Inplace copy: copy_(dst, src) -> dst  →  _foreach_copy_(dsts, srcs)
_COPY_OP = aten.copy_.default

_MIN_GROUP_SIZE = 3


def _get_binary_key(node: fx.Node) -> tuple | None:
    if node.op != "call_function" or node.target not in _FUSEABLE_BINARY_OPS:
        return None

    args = node.args
    if len(args) < 2:
        return None

    first_arg = args[0]
    if not isinstance(first_arg, fx.Node):
        return None
    val = first_arg.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None

    second_arg = args[1]
    if isinstance(second_arg, (int, float)):
        other_key = ("scalar", second_arg)
    elif isinstance(second_arg, fx.Node):
        other_val = second_arg.meta.get("val")
        if isinstance(other_val, torch.Tensor):
            other_key = ("tensor", tuple(other_val.shape), str(other_val.dtype))
        else:
            return None
    else:
        return None

    alpha = node.kwargs.get("alpha", 1.0)

    return (
        node.target,
        tuple(val.shape),
        str(val.dtype),
        other_key,
        alpha,
    )


def _get_unary_key(node: fx.Node) -> tuple | None:
    if node.op != "call_function" or node.target not in _FUSEABLE_UNARY_OPS:
        return None

    args = node.args
    if len(args) < 1:
        return None

    first_arg = args[0]
    if not isinstance(first_arg, fx.Node):
        return None
    val = first_arg.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None

    return (
        node.target,
        tuple(val.shape),
        str(val.dtype),
    )


def _get_copy_key(node: fx.Node) -> tuple | None:
    if node.op != "call_function" or node.target is not _COPY_OP:
        return None

    args = node.args
    if len(args) < 2:
        return None

    dst, src = args[0], args[1]
    if not isinstance(dst, fx.Node) or not isinstance(src, fx.Node):
        return None

    src_val = src.meta.get("val")
    if not isinstance(src_val, torch.Tensor):
        return None

    return (
        _COPY_OP,
        tuple(src_val.shape),
        str(src_val.dtype),
        str(src_val.device),
    )


def _flood_forward(start: fx.Node, visited: set[fx.Node]) -> None:
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(cur.users.keys())


def _are_independent(nodes: list[fx.Node]) -> list[fx.Node]:
    """Return the largest independent subset.

    For a foreach op, ALL input tensors go into a single call. So no input
    tensor can transitively depend on any other input tensor.
    """
    if len(nodes) <= 1:
        return nodes

    selected: list[fx.Node] = []
    selected_forward: set[fx.Node] = set()
    selected_backward: set[fx.Node] = set()

    def _flood(start: fx.Node, visited: set[fx.Node], forward: bool) -> None:
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(cur.users.keys() if forward else cur.all_input_nodes)

    for node in nodes:
        if node in selected_forward or node in selected_backward:
            continue
        all_inputs = set(node.all_input_nodes)
        if all_inputs & selected_forward or all_inputs & selected_backward:
            continue

        selected.append(node)
        _flood(node, selected_forward, forward=True)
        _flood(node, selected_backward, forward=False)
        for inp in node.all_input_nodes:
            _flood(inp, selected_forward, forward=True)
            _flood(inp, selected_backward, forward=False)

    return selected


def _find_independent_subsets(
    nodes: list[fx.Node], min_size: int
) -> list[list[fx.Node]]:
    """Partition nodes into independent subsets using bidirectional reachability.

    A foreach op merges all selected nodes' inputs into one call and produces
    all outputs. Any path between selected nodes (forward OR backward, including
    through their inputs) would create a cycle. We use _are_independent which
    does full bidirectional flood fill.
    """
    if len(nodes) < min_size:
        return []

    batch = _are_independent(nodes)
    if len(batch) >= min_size:
        return [batch]
    return []


def _fuse_binary_group(
    graph: fx.Graph, key: tuple, nodes: list[fx.Node]
) -> None:
    target = key[0]
    foreach_scalar_op, foreach_list_op = _FUSEABLE_BINARY_OPS[target]

    self_tensors = [n.args[0] for n in nodes]
    other_args = [n.args[1] for n in nodes]

    first_other = other_args[0]
    all_same_scalar = all(
        isinstance(o, (int, float)) and o == first_other for o in other_args
    )

    with graph.inserting_before(nodes[-1]):
        if all_same_scalar:
            result = graph.call_function(
                foreach_scalar_op, args=(self_tensors, first_other)
            )
        else:
            result = graph.call_function(
                foreach_list_op, args=(self_tensors, other_args)
            )

        result.meta["val"] = [n.meta.get("val") for n in nodes]

        for i, node in enumerate(nodes):
            with graph.inserting_after(result):
                getitem = graph.call_function(
                    operator.getitem, args=(result, i)
                )
                getitem.meta.update(node.meta)
            node.replace_all_uses_with(getitem)

    for node in nodes:
        graph.erase_node(node)

    log.info(
        "horizontal_fusion: fused %d %s ops into %s",
        len(nodes),
        target.__name__,
        foreach_scalar_op.__name__ if all_same_scalar else foreach_list_op.__name__,
    )


def _fuse_unary_group(
    graph: fx.Graph, key: tuple, nodes: list[fx.Node]
) -> None:
    target = key[0]
    foreach_op = _FUSEABLE_UNARY_OPS[target]

    self_tensors = [n.args[0] for n in nodes]

    with graph.inserting_before(nodes[-1]):
        result = graph.call_function(foreach_op, args=(self_tensors,))
        result.meta["val"] = [n.meta.get("val") for n in nodes]

        for i, node in enumerate(nodes):
            with graph.inserting_after(result):
                getitem = graph.call_function(
                    operator.getitem, args=(result, i)
                )
                getitem.meta.update(node.meta)
            node.replace_all_uses_with(getitem)

    for node in nodes:
        graph.erase_node(node)

    log.info(
        "horizontal_fusion: fused %d %s ops into %s",
        len(nodes),
        target.__name__,
        foreach_op.__name__,
    )


def _get_mm_key(node: fx.Node) -> tuple | None:
    if node.op != "call_function" or node.target is not aten.mm.default:
        return None

    args = node.args
    if len(args) != 2:
        return None

    lhs, rhs = args
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
        return None

    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return None

    if any(isinstance(s, torch.SymInt) for s in lhs_val.shape) or any(
        isinstance(s, torch.SymInt) for s in rhs_val.shape
    ):
        return None

    return (
        aten.mm.default,
        tuple(lhs_val.shape),
        tuple(rhs_val.shape),
        lhs_val.dtype,
        str(lhs_val.device),
    )


def _fuse_mm_group(
    graph: fx.Graph, nodes: list[fx.Node]
) -> None:
    lhs_list = [n.args[0] for n in nodes]
    rhs_list = [n.args[1] for n in nodes]
    n_ops = len(nodes)
    lhs_shape = tuple(lhs_list[0].meta["val"].shape)
    rhs_shape = tuple(rhs_list[0].meta["val"].shape)

    with graph.inserting_before(nodes[-1]):
        result = graph.call_function(
            aten._foreach_mm.default, args=(lhs_list, rhs_list)
        )
        result.meta["val"] = [
            aten.mm(l.meta["val"], r.meta["val"])
            for l, r in zip(lhs_list, rhs_list)
        ]

        for i, node in enumerate(nodes):
            with graph.inserting_after(result):
                getitem = graph.call_function(
                    operator.getitem, args=(result, i)
                )
                getitem.meta.update(node.meta)
            node.replace_all_uses_with(getitem)

    for node in nodes:
        graph.erase_node(node)

    log.info(
        "horizontal_fusion: fused %d mm ops (%s × %s) into _foreach_mm",
        n_ops,
        lhs_shape,
        rhs_shape,
    )


def _fuse_copy_group(
    graph: fx.Graph, nodes: list[fx.Node]
) -> None:
    """Fuse copy_(dst, src) nodes into _foreach_copy_(dsts, srcs).

    copy_ returns self (the dst tensor after mutation). After
    _foreach_copy_ mutates all dsts in place, we replace each original
    copy_ result with its dst arg which now holds the new value.
    """
    dsts = [n.args[0] for n in nodes]
    srcs = [n.args[1] for n in nodes]

    with graph.inserting_before(nodes[-1]):
        foreach_copy = graph.call_function(
            aten._foreach_copy_.default, args=(dsts, srcs)
        )
        foreach_copy.meta["val"] = None

    for node, dst in zip(nodes, dsts):
        node.replace_all_uses_with(dst)

    for node in reversed(nodes):
        graph.erase_node(node)

    log.info(
        "horizontal_fusion: fused %d copy_ ops into _foreach_copy_",
        len(nodes),
    )


def foreach_mm_preorder(gm: torch.fx.GraphModule) -> bool:
    """Pack independent same-shape mm ops together.

    Uses forward-only flood (not bidirectional) to find independent groups.
    Bidirectional flood is too conservative — it rejects ops that share a
    common ancestor (e.g., all params coming from the same all_gather bucket).
    Forward-only correctly identifies ops as independent when neither's output
    reaches the other.

    Iteratively extracts batches: first batch at one NS iteration level,
    second batch at the next level, etc.
    """
    graph = gm.graph
    changed = False

    mm_groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    for node in graph.nodes:
        key = _get_mm_key(node)
        if key is not None:
            mm_groups[key].append(node)

    for key, nodes in mm_groups.items():
        # Use forward-only flood to find independent subsets iteratively
        for batch in _find_independent_subsets_forward(nodes, _MIN_GROUP_SIZE):
            # Move all batch nodes to be adjacent, just before the
            # earliest user of any batch node
            batch_set = set(batch)
            earliest_user = None
            for node in graph.nodes:
                if node in batch_set:
                    continue
                for inp in node.all_input_nodes:
                    if inp in batch_set:
                        earliest_user = node
                        break
                if earliest_user is not None:
                    break

            if earliest_user is None:
                continue

            for node in batch:
                node.prepend(earliest_user)

            changed = True
            log.info(
                "foreach_mm_preorder: packed %d mm ops (%s x %s) together",
                len(batch), key[1], key[2],
            )

    if changed:
        from torch._inductor.pattern_matcher import stable_topological_sort

        try:
            stable_topological_sort(graph)
            graph.lint()
        except AssertionError:
            log.warning("foreach_mm_preorder: topo sort failed, skipping")
            return False

    return changed


def _find_independent_subsets_forward(
    nodes: list[fx.Node], min_size: int
) -> list[list[fx.Node]]:
    """Find independent subsets using FORWARD-ONLY flood.

    Unlike _are_independent (bidirectional), this only checks forward
    reachability. Two nodes are independent if neither is in the other's
    forward closure. Shared ancestors (like all_gather buckets) don't
    cause false dependencies.
    """
    if len(nodes) < min_size:
        return []

    batches: list[list[fx.Node]] = []
    remaining = list(nodes)

    while len(remaining) >= min_size:
        selected: list[fx.Node] = []
        blocked: set[fx.Node] = set()
        deferred: list[fx.Node] = []

        for node in remaining:
            if node in blocked:
                deferred.append(node)
                continue
            selected.append(node)
            _flood_forward(node, blocked)

        if len(selected) >= min_size:
            batches.append(selected)
        remaining = deferred

    return batches


def horizontal_fusion_pass(gm: torch.fx.GraphModule) -> bool:
    """Fuse repeated identical ops as _foreach_* calls.

    Handles mm (when foreach_mm enabled), binary (mul, add, sub, div, pow,
    clamp_min), unary (sqrt, neg), and copy_ ops.

    mm is fused FIRST while per-parameter subgraphs are still independent.
    Elementwise fusions run after and create cross-parameter dependencies
    that would block mm batching if done in the other order.
    """
    from .. import config

    foreach_mm_enabled = config.aten_distributed_optimizations.foreach_mm
    foreach_mm_min = getattr(
        config.aten_distributed_optimizations,
        "foreach_mm_min_group_size",
        _MIN_GROUP_SIZE,
    )

    graph = gm.graph
    changed = False

    binary_groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    unary_groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    copy_groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    mm_groups: dict[tuple, list[fx.Node]] = defaultdict(list)

    for node in graph.nodes:
        if foreach_mm_enabled:
            key = _get_mm_key(node)
            if key is not None:
                mm_groups[key].append(node)
                continue
        key = _get_binary_key(node)
        if key is not None:
            binary_groups[key].append(node)
            continue
        key = _get_unary_key(node)
        if key is not None:
            unary_groups[key].append(node)
            continue
        key = _get_copy_key(node)
        if key is not None:
            copy_groups[key].append(node)

    # mm FIRST — parameters are independent at this point
    for key, nodes in mm_groups.items():
        for batch in _find_independent_subsets(nodes, foreach_mm_min):
            _fuse_mm_group(graph, batch)
            changed = True

    for key, nodes in binary_groups.items():
        for batch in _find_independent_subsets(nodes, _MIN_GROUP_SIZE):
            _fuse_binary_group(graph, key, batch)
            changed = True

    for key, nodes in unary_groups.items():
        for batch in _find_independent_subsets(nodes, _MIN_GROUP_SIZE):
            _fuse_unary_group(graph, key, batch)
            changed = True

    for key, nodes in copy_groups.items():
        for batch in _find_independent_subsets(nodes, _MIN_GROUP_SIZE):
            _fuse_copy_group(graph, batch)
            changed = True

    if changed:
        from torch._inductor.pattern_matcher import stable_topological_sort

        try:
            stable_topological_sort(graph)
            graph.eliminate_dead_code()
            graph.lint()
        except AssertionError:
            log.warning("horizontal_fusion: topo sort failed, graph has cycles — skipping")
            raise

    return changed


def foreach_mm_pass(gm: torch.fx.GraphModule) -> bool:
    """Batch independent same-shape mm calls into _foreach_mm.

    Must run AFTER overlap scheduling so the scheduler sees individual mm
    nodes and can interleave them with collectives for comm/compute overlap.
    """
    from .. import config

    min_size = getattr(
        config.aten_distributed_optimizations,
        "foreach_mm_min_group_size",
        _MIN_GROUP_SIZE,
    )

    graph = gm.graph
    changed = False

    mm_groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    for node in graph.nodes:
        key = _get_mm_key(node)
        if key is not None:
            mm_groups[key].append(node)

    for key, nodes in mm_groups.items():
        for batch in _find_independent_subsets(nodes, min_size):
            _fuse_mm_group(graph, batch)
            changed = True

    if changed:
        from torch._inductor.pattern_matcher import stable_topological_sort

        try:
            stable_topological_sort(graph)
            graph.eliminate_dead_code()
            graph.lint()
        except AssertionError:
            log.warning("foreach_mm: topo sort failed — skipping")
            raise

    return changed


