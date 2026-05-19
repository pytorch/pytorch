import math
from typing import Any

import sympy

import torch
from torch.fx.node import map_arg
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .loop_body import LoopBody
from .ops_handler import OP_NAMES
from .utils import dominated_nodes


def val_expressable_in_32_bits(val: Any) -> bool:
    if getattr(val, "is_Boolean", False):
        return True

    if isinstance(val, sympy.Expr):
        assert val.is_number
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)

    # bound within mantissa
    if isinstance(val, float):
        return val <= (2**24) and val >= -(2**24)

    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min

    raise TypeError(f"Unexpected value {val}")


def range_expressable_in_32_bits(range: ValueRanges[sympy.Expr]) -> bool:
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(
        range.upper
    )


def _dominated_uses_fit_in_32_bits(
    node: torch.fx.Node,
    bounds: dict[Any, ValueRanges[Any]],
    indirect_vars: list[Any],
    indices: dict[Any, sympy.Expr],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
    value_use: OrderedSet[torch.fx.Node] | None = None,
) -> bool:
    # If a downstream use explicitly converts to int32 or float, precision is
    # fixed for that chain and we do not need to inspect dominated values past it.
    def skip_filter(user: torch.fx.Node) -> bool:
        return user.target == "to_dtype" and user.args[2] in (
            torch.int32,
            torch.float32,
            torch.float64,
        )

    for dominated in dominated_nodes([node], skip_filter):
        if value_use is not None and dominated not in value_use:
            continue

        if dominated.target in ["store", "output"]:
            continue

        if isinstance(dominated.target, str) and "set_indirect" in dominated.target:
            idx = int(dominated.target[len("set_indirect") :])
            indirect_var = indirect_vars[idx]

            # We check that we can compute all the indices it's involved in with int32
            for index, expr in indices.items():
                if indirect_var in expr.free_symbols:
                    index_val = replacement_vals[index]

                    if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                        return False

                    # all indices are integers, so make sure that we
                    # use the bounds of integers instead of floats.
                    # TODO - not sure if we should be doing int/float casts while tracing,
                    # might interfere with sympy.

                    index_val_int = ValueRanges[sympy.Expr](
                        int(index_val.lower), int(index_val.upper)
                    )
                    if not range_expressable_in_32_bits(index_val_int):
                        return False

        dominated_bounds = bounds.get(dominated)
        if dominated_bounds is None or not range_expressable_in_32_bits(
            dominated_bounds
        ):
            return False

    return True


def try_to_reduce_precision(
    node: Any,
    bounds: dict[Any, Any],
    indirect_vars: list[Any],
    indices: dict[Any, sympy.Expr],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
) -> None:
    # TODO - there are dominated uses whose dtype does not depend on whether
    # we reduce the precision here, e.g. add(int64, int64) one of the args can be reduced to
    # int32 without changing the output precision of the node. this case hasn't shown up
    if not _dominated_uses_fit_in_32_bits(
        node, bounds, indirect_vars, indices, replacement_vals
    ):
        return

    args = list(node.args)
    args[2] = torch.int32
    node.args = tuple(args)


def indexing_dtype_strength_reduction(loop_body: LoopBody) -> None:
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
    bv = loop_body.bounds()

    int64_dtype_nodes = [
        node
        for node in loop_body.get_nodes()
        if (
            node.target == "to_dtype"
            and node.args[2] == torch.int64
            and node not in bv.unbounded_vars
        )
    ]
    if not int64_dtype_nodes:
        return

    bounds = bv.get_bounds()

    # TODO - if dominated node of one to_dtype is not expressible in int32,
    # we should short circuit another to_dtype node if that node also dominates
    for node in int64_dtype_nodes:
        try_to_reduce_precision(
            node,
            bounds,
            loop_body.indirect_vars,
            loop_body.indexing_exprs,
            bv.replacement_vals,
        )


# Op targets in a LoopBody FX graph that act as terminal sinks for either
# indexing computations or value computations.
_INDEXING_SINK_ARGS: dict[str, tuple[int, ...]] = {
    "load": (2,),
    "store": (2,),
    "store_reduction": (2,),
    "check_bounds": (1, 2),
    "bucketize": (2, 3, 6, 7),
}
_VALUE_SINK_ARGS: dict[str, tuple[int, ...]] = {
    "store": (3,),
    "store_reduction": (3,),
    "reduction": (4,),
    "partial_accumulate": (3,),
    "scan": (3,),
    "sort": (2,),
    "bucketize": (1,),
}

_NON_VALUE_PROPAGATING_TARGETS: OrderedSet[str] = OrderedSet(
    [
        "check_bounds",
        "device_assert_async",
        "indirect_indexing",
        "load",
        "load_seed",
        "masked",
        "output",
        "placeholder",
    ]
)
_VALUE_PROPAGATING_TARGETS = (
    OP_NAMES
    - OrderedSet(_INDEXING_SINK_ARGS)
    - OrderedSet(_VALUE_SINK_ARGS)
    - _NON_VALUE_PROPAGATING_TARGETS
)


def _is_masked_subblock(node: torch.fx.Node) -> bool:
    return isinstance(node.target, str) and "masked_subblock" in node.target


def _loop_body_graphs(loop_body: LoopBody) -> list[torch.fx.Graph]:
    return [
        loop_body.root_block.graph,
        *(block.graph for block in getattr(loop_body, "subblocks", {}).values()),
    ]


def _collect_index_value_sinks(
    graph: torch.fx.Graph,
) -> tuple[OrderedSet[torch.fx.Node], OrderedSet[torch.fx.Node]]:
    """
    Classify the FX node arguments by usage at terminal sinks.

    Returns (indexing_sinks, value_sinks): nodes whose result feeds an indexing
    position (load/store index, check_bounds, indirect indexing) vs. a value
    position (store value, reduction value, output).
    """
    indexing_sinks: OrderedSet[torch.fx.Node] = OrderedSet()
    value_sinks: OrderedSet[torch.fx.Node] = OrderedSet()

    def _add(target: OrderedSet[torch.fx.Node], arg: Any) -> None:
        def add_node(n: torch.fx.Node) -> torch.fx.Node:
            target.add(n)
            return n

        map_arg(arg, add_node)

    for node in graph.nodes:
        if node.op == "output":
            for a in node.args:
                _add(value_sinks, a)
            continue

        target = node.target
        if isinstance(target, str):
            for idx in _INDEXING_SINK_ARGS.get(target, ()):
                if idx < len(node.args):
                    _add(indexing_sinks, node.args[idx])
            for idx in _VALUE_SINK_ARGS.get(target, ()):
                if idx < len(node.args):
                    _add(value_sinks, node.args[idx])
        if node.op == "call_module" and isinstance(target, str):
            if target.startswith("set_"):
                # set_indirect_<n>: argument flows into indirect indexing.
                for a in node.args:
                    _add(indexing_sinks, a)
            elif "masked_subblock" in target:
                # masked_subblock(mask, other): the mask controls execution and
                # should stay on the indexing/control path; only `other` is a
                # value input to the subblock result.
                if len(node.args) > 0:
                    _add(indexing_sinks, node.args[0])
                if len(node.args) > 1:
                    _add(value_sinks, node.args[1])
            elif target.startswith("scan"):
                if len(node.args) > 1:
                    _add(value_sinks, node.args[1])

    return indexing_sinks, value_sinks


def _mark_ancestors(
    starts: OrderedSet[torch.fx.Node],
    *,
    value_flow: bool = False,
) -> OrderedSet[torch.fx.Node]:
    """
    Walk backward from `starts`, accumulating ancestors. `load` is a barrier:
    its output value is determined by what is at the loaded address, not by
    the index expression, so usage does not propagate into the load's index.

    Value flow is intentionally conservative: only known value-propagating ops
    are transparent. Unknown targets are treated as indexing/control use.
    """
    marked: OrderedSet[torch.fx.Node] = OrderedSet()
    stack: list[torch.fx.Node] = list(starts)
    while stack:
        node = stack.pop()
        if node in marked:
            continue
        marked.add(node)
        target = node.target
        if target == "load" or _is_masked_subblock(node):
            continue
        if value_flow and target not in _VALUE_PROPAGATING_TARGETS:
            continue

        def append_node(n: torch.fx.Node) -> torch.fx.Node:
            stack.append(n)
            return n

        map_arg(node.args, append_node)
        map_arg(node.kwargs, append_node)
    return marked


def _compute_value_expr_dtype(
    loop_body: LoopBody,
    node: torch.fx.Node,
    bounds: dict[torch.fx.Node, ValueRanges[Any]],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
    value_use: OrderedSet[torch.fx.Node],
) -> torch.dtype | None:
    dtype_arg = node.args[2] if len(node.args) > 2 else None
    if not isinstance(dtype_arg, torch.dtype):
        return None
    dtype: torch.dtype = dtype_arg
    if dtype == torch.int32:
        return torch.int32

    get_index_node = node.args[1] if len(node.args) > 1 else None
    if (
        not isinstance(get_index_node, torch.fx.Node)
        or get_index_node.target != "get_index"
    ):
        return None
    index_name = get_index_node.args[0]
    assert isinstance(index_name, str)
    sympy_expr = loop_body.indexing_exprs.get(index_name)
    if not isinstance(sympy_expr, sympy.Expr):
        return None

    if not sympy_expr.is_integer:
        return dtype

    if dtype == torch.bool:
        return dtype

    if dtype not in (torch.int32, torch.int64):
        if range_expressable_in_32_bits(bound_sympy(sympy_expr, replacement_vals)):
            return dtype
        return torch.int64

    requires_int64 = not _dominated_uses_fit_in_32_bits(
        node,
        bounds,
        loop_body.indirect_vars,
        loop_body.indexing_exprs,
        replacement_vals,
        value_use=value_use,
    )
    if dtype == torch.int64:
        return torch.int64 if requires_int64 else torch.int32
    return dtype


def _rewrite_value_expr_dtype(
    loop_body: LoopBody,
    node: torch.fx.Node,
    bounds: dict[torch.fx.Node, ValueRanges[Any]],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
    value_use: OrderedSet[torch.fx.Node],
) -> None:
    dtype = _compute_value_expr_dtype(
        loop_body, node, bounds, replacement_vals, value_use
    )
    if dtype is not None:
        args = list(node.args)
        args[2] = dtype
        node.args = tuple(args)


def convert_index_expr_to_value_expr(loop_body: LoopBody) -> None:
    """
    Rewrite value uses of ``index_expr`` FX nodes to ``value_expr``. This lets
    codegen honor the requested dtype for value uses (where the kernel's int32
    narrowing of ``index_expr`` would silently drop precision) without
    affecting genuine indexing uses.

    The classification is purely structural / data-flow:
      - Walk backward from value sinks (store value, reduction value,
        output) and indexing sinks (load index, store index, check_bounds,
        set_indirect), with ``load`` as a barrier (its output value is
        determined by what the load reads, not the index expression).
      - Any ``index_expr`` reachable from a value sink is rewritten to
        ``value_expr`` on the value path. If the same node also reaches an
        indexing sink, the value path is cloned first so indexing uses keep
        the original ``index_expr``.

    This is a safety net for legacy lowering callsites that still emit
    ``index_expr`` for value uses. New lowerings should prefer to emit
    ``value_expr`` directly when their intent is a value computation.
    """
    graphs = _loop_body_graphs(loop_body)
    index_expr_nodes = [
        node for graph in graphs for node in graph.nodes if node.target == "index_expr"
    ]
    if not index_expr_nodes:
        return

    def rewrite_graph(graph: torch.fx.Graph) -> None:
        indexing_sinks, value_sinks = _collect_index_value_sinks(graph)
        indexing_use = _mark_ancestors(indexing_sinks)
        value_use = _mark_ancestors(value_sinks, value_flow=True)
        value_clones: dict[torch.fx.Node, torch.fx.Node] = {}

        def value_version(node: torch.fx.Node, anchor: torch.fx.Node) -> torch.fx.Node:
            # Value-only chains can be rewritten in place. Mixed value/indexing
            # chains are cloned so indexing users keep the original path.
            if node.op == "placeholder":
                return node
            if node.target == "load" or _is_masked_subblock(node):
                return node
            if node.target == "get_index" or node not in value_use:
                return node
            if node.target == "index_expr":
                if node not in indexing_use:
                    node.target = "value_expr"
                    return node
                if node not in value_clones:
                    with graph.inserting_before(anchor):
                        clone = graph.call_method(
                            "value_expr", node.args, dict(node.kwargs)
                        )
                    clone.meta = node.meta.copy()
                    value_clones[node] = clone
                return value_clones[node]

            if node not in indexing_use:
                node.args = map_arg(node.args, lambda n: value_version(n, node))
                node.kwargs = map_arg(node.kwargs, lambda n: value_version(n, node))
                return node

            if node not in value_clones:
                with graph.inserting_before(anchor):
                    clone = graph.node_copy(node, lambda n: value_version(n, anchor))
                clone.meta = node.meta.copy()
                value_clones[node] = clone
            return value_clones[node]

        def rewrite_value_arg(arg: Any, anchor: torch.fx.Node) -> Any:
            return map_arg(arg, lambda n: value_version(n, anchor))

        for node in graph.nodes:
            if node.op == "output":
                node.args = rewrite_value_arg(node.args, node)
                continue

            if not isinstance(node.target, str):
                continue

            value_arg_indices = _VALUE_SINK_ARGS.get(node.target, ())
            if node.op == "call_module":
                if _is_masked_subblock(node) or node.target.startswith("scan"):
                    value_arg_indices = (1,)
            if not value_arg_indices:
                continue
            args = list(node.args)
            for idx in value_arg_indices:
                if idx < len(args):
                    args[idx] = rewrite_value_arg(args[idx], node)
            node.args = tuple(args)

        graph.lint()

    for graph in graphs:
        rewrite_graph(graph)

    bound_vars = loop_body.bounds()
    bounds = bound_vars.get_bounds()
    for graph in graphs:
        _, value_sinks = _collect_index_value_sinks(graph)
        value_use = _mark_ancestors(value_sinks, value_flow=True)
        for node in graph.nodes:
            if node.target == "value_expr":
                _rewrite_value_expr_dtype(
                    loop_body, node, bounds, bound_vars.replacement_vals, value_use
                )
        graph.lint()
