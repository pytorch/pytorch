import math
from typing import Any

import sympy

import torch
from torch.fx.node import map_arg
from torch.utils._sympy.value_ranges import ValueRanges

from .loop_body import LoopBody
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


def try_to_reduce_precision(
    node: Any,
    bounds: dict[Any, Any],
    indirect_vars: list[Any],
    indices: dict[Any, sympy.Expr],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
) -> None:
    # if a downstream use of a node explicitly converts to int32, or float16/float32/float64,
    # then it's precision is set for that chain of uses, and we don't need to consider those
    # dominated values
    def skip_filter(node: Any) -> bool:
        return node.target == "to_dtype" and node.args[2] in (
            torch.int32,
            torch.float32,
            torch.float64,
        )

    # TODO - there are dominated uses whose dtype does not depend on whether
    # we reduce the precision here, e.g. add(int64, int64) one of the args can be reduced to
    # int32 without changing the output precision of the node. this case hasn't shown up
    for dominated in dominated_nodes([node], skip_filter):
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
                        return

                    # all indices are integers, so make sure that we
                    # use the bounds of integers instead of floats.
                    # TODO - not sure if we should be doing int/float casts while tracing,
                    # might interfere with sympy.

                    index_val_int = ValueRanges[sympy.Expr](
                        int(index_val.lower), int(index_val.upper)
                    )
                    if not range_expressable_in_32_bits(index_val_int):
                        return

        if not range_expressable_in_32_bits(bounds[dominated]):
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


def _is_value_sink_arg(node: torch.fx.Node, arg_idx: int) -> bool:
    """True if ``node.args[arg_idx]`` is a value sink (feeds tensor output)."""
    target = node.target
    # call_method args are (ops, *real_args), so real arg indices are shifted by 1
    if node.op == "call_method":
        # store(ops, name, index, value, mode) — value is real arg 2 → args[3]
        if target in ("store", "store_reduction"):
            return arg_idx == 3
        # reduction(ops, dtype, src_dtype, reduction_type, value) — value is args[4]
        if target == "reduction":
            return arg_idx == 4
        # partial_accumulate(ops, name, reduction_type, value, ...) — value is args[3]
        if target == "partial_accumulate":
            return arg_idx == 3
        # sort(ops, dtypes, values, ...) — values is args[2]
        if target == "sort":
            return arg_idx == 2
        # scan(ops, dtypes, combine_fn, values) — values is args[3]
        if target == "scan":
            return arg_idx == 3
    if node.op == "call_module" and isinstance(target, str):
        # scan_subblock(dtypes, values) — values is args[1]
        if target.startswith("scan"):
            return arg_idx == 1
        # masked_subblock(mask, other) — other is args[1]
        if target.startswith("masked_subblock"):
            return arg_idx == 1
    return False


_VALUE_PROPAGATING_OPS: set[str] = {
    "abs",
    "add",
    "and_",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "ceil",
    "ceil_to_int",
    "clamp_max",
    "clamp_min",
    "constant",
    "cos",
    "cosh",
    "div_rn",
    "dot",
    "eq",
    "erf",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "floor_to_int",
    "floordiv",
    "fmod",
    "frexp",
    "ge",
    "getitem",
    "gt",
    "identity",
    "index_expr",
    "int_truediv",
    "isinf",
    "isnan",
    "le",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "lshift",
    "lt",
    "maximum",
    "minimum",
    "mod",
    "mul",
    "ne",
    "neg",
    "not_",
    "or_",
    "pow",
    "reciprocal",
    "remainder",
    "round",
    "round_to_int",
    "rsqrt",
    "rshift",
    "sigmoid",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "tan",
    "tanh",
    "to_dtype",
    "to_dtype_bitcast",
    "truediv",
    "trunc",
    "trunc_to_int",
    "truncdiv",
    "value_expr",
    "where",
    "xor_",
}


def _is_value_propagating(node: torch.fx.Node) -> bool:
    if node.op in ("placeholder", "call_function"):
        return True
    if node.op == "call_module":
        return True
    if node.op != "call_method":
        return False
    return node.target in _VALUE_PROPAGATING_OPS


def _find_value_use(graph: torch.fx.Graph) -> set[torch.fx.Node]:
    """
    Walk backward from value sinks through value-propagating ops.
    Returns the set of nodes reachable from value computation.
    """
    value_use: set[torch.fx.Node] = set()
    stack: list[torch.fx.Node] = []

    def _add_node(n: torch.fx.Node) -> torch.fx.Node:
        stack.append(n)
        return n

    def _add_arg(arg: Any) -> None:
        map_arg(arg, _add_node)

    for node in graph.nodes:
        if node.op == "output":
            _add_arg(node.args)
            continue
        for i, arg in enumerate(node.args):
            if _is_value_sink_arg(node, i):
                _add_arg(arg)

    while stack:
        node = stack.pop()
        if node in value_use:
            continue
        value_use.add(node)
        if not _is_value_propagating(node):
            continue
        _add_arg(node.args)
        _add_arg(node.kwargs)

    return value_use


def convert_index_expr_to_value_expr(loop_body: LoopBody) -> None:
    """
    Rewrite ``index_expr`` nodes that participate in value computation to
    ``value_expr``, so codegen can honor the requested dtype instead of
    narrowing to the kernel's indexing dtype.

    Classification: walk backward from value sinks (store value, reduction,
    output) and indexing sinks (load index, store index, check_bounds,
    set_indirect) with ``load`` as a barrier. Any ``index_expr`` reachable
    from a value sink is converted in-place to ``value_expr``.

    TODO: if mixed use (same index_expr used for both indexing and value)
    ever occurs in practice, the node should be cloned so the indexing path
    keeps the original index_expr. This hasn't been observed so far.
    """
    graphs = [
        loop_body.root_block.graph,
        *(block.graph for block in getattr(loop_body, "subblocks", {}).values()),
    ]
    if not any(
        graph.find_nodes(op="call_method", target="index_expr", sort=False)
        for graph in graphs
    ):
        return

    for graph in graphs:
        value_use = _find_value_use(graph)
        for node in graph.nodes:
            if node.target == "index_expr" and node in value_use:
                node.target = "value_expr"

    LoopBody.get_nodes.clear_cache(loop_body)
    LoopBody.bounds.clear_cache(loop_body)
