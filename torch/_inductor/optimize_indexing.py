import math
import operator
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.fx.node import map_aggregate, map_arg
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


@dataclass
class _RuleArg:
    value: Any


@dataclass(frozen=True)
class _IndexValueRule:
    # Inputs in the LoopBody FX node that act as terminal sinks.
    indexing_sinks: tuple[Any, ...] = ()
    value_sinks: tuple[Any, ...] = ()
    # Inputs that receive value use from this op's result. None means all inputs.
    value_inputs: tuple[Any, ...] | None = ()


@dataclass(frozen=True)
class _BoundIndexValueRule:
    rule: _IndexValueRule
    args: tuple[_RuleArg, ...]
    kwargs: dict[str, _RuleArg]


class _IndexValueOpsHandler:
    """
    Classify each OpsHandler op for index/value-use analysis.
    """

    _VALUE_PROPAGATING_OPS = (
        "abs",
        "acos",
        "acosh",
        "add",
        "airy_ai",
        "and_",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "bessel_j0",
        "bessel_j1",
        "bessel_y0",
        "bessel_y1",
        "bitwise_and",
        "bitwise_left_shift",
        "bitwise_not",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "ceil",
        "ceil_to_int",
        "chebyshev_polynomial_t",
        "chebyshev_polynomial_u",
        "chebyshev_polynomial_v",
        "chebyshev_polynomial_w",
        "constant",
        "copysign",
        "cos",
        "cosh",
        "digamma",
        "div_rn",
        "dot",
        "eq",
        "erf",
        "erfc",
        "erfcx",
        "erfinv",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "floor_to_int",
        "floordiv",
        "fma",
        "fmod",
        "frexp",
        "gammainc",
        "gammaincc",
        "ge",
        "gt",
        "halide_clamp",
        "hermite_polynomial_h",
        "hermite_polynomial_he",
        "hypot",
        "i0",
        "i0e",
        "i1",
        "i1e",
        "identity",
        "igamma",
        "igammac",
        "index_expr",
        "inline_asm_elementwise",
        "int_truediv",
        "isinf",
        "isnan",
        "laguerre_polynomial_l",
        "ldexp",
        "le",
        "legendre_polynomial_p",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "log_ndtr",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "lshift",
        "lt",
        "maximum",
        "minimum",
        "mod",
        "modified_bessel_i0",
        "modified_bessel_i1",
        "modified_bessel_k0",
        "modified_bessel_k1",
        "mul",
        "mul_rn",
        "ndtr",
        "ndtri",
        "ne",
        "neg",
        "nextafter",
        "or_",
        "polygamma",
        "pow",
        "rand",
        "rand_eager",
        "randint64",
        "randn",
        "reciprocal",
        "relu",
        "remainder",
        "round",
        "round_to_int",
        "rshift",
        "rsqrt",
        "scaled_modified_bessel_k0",
        "scaled_modified_bessel_k1",
        "shifted_chebyshev_polynomial_t",
        "shifted_chebyshev_polynomial_u",
        "shifted_chebyshev_polynomial_v",
        "shifted_chebyshev_polynomial_w",
        "sigmoid",
        "sign",
        "signbit",
        "sin",
        "sinh",
        "spherical_bessel_j0",
        "sqrt",
        "square",
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
        "xor",
        "zeta",
    )
    _EXPLICIT_RULE_OPS = (
        "bucketize",
        "check_bounds",
        "device_assert_async",
        "indirect_indexing",
        "load",
        "load_seed",
        "masked",
        "output",
        "partial_accumulate",
        "placeholder",
        "reduction",
        "scan",
        "sort",
        "store",
        "store_reduction",
    )

    def __init__(self) -> None:
        self._install_bulk_rules()
        classified_ops = OrderedSet(self._VALUE_PROPAGATING_OPS) | OrderedSet(
            self._EXPLICIT_RULE_OPS
        )
        unknown_ops = classified_ops - OP_NAMES
        torch._check(
            len(unknown_ops) == 0,
            lambda: f"Value/index rules for unknown ops: {unknown_ops}",
        )
        unimplemented_ops = OP_NAMES - classified_ops
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented value/index rule for ops: {unimplemented_ops}",
        )

    def rule_for_node(self, node: torch.fx.Node) -> _BoundIndexValueRule | None:
        if node.op == "output":
            return self._bind_rule(self.output, node.args, node.kwargs)
        if node.op == "call_function" and node.target is operator.getitem:
            return self._bind_rule(self.getitem, node.args, node.kwargs)
        if not isinstance(node.target, str):
            return None

        target = node.target
        if node.op == "call_module":
            if target.startswith("set_"):
                return self._bind_rule(self.set_indirect, node.args, node.kwargs)
            if target.startswith("masked_subblock"):
                return self._bind_rule(self.masked_subblock, node.args, node.kwargs)
            if target.startswith("scan"):
                return self._bind_rule(self.scan_subblock, node.args, node.kwargs)
            return None

        if target not in OP_NAMES:
            return None

        args = node.args
        if node.op == "call_method":
            args = args[1:]
        return self._bind_rule(
            getattr(self, target),
            args,
            node.kwargs,
        )

    def _bind_rule(
        self,
        rule_fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _BoundIndexValueRule:
        rule_args = tuple(_RuleArg(arg) for arg in args)
        rule_kwargs = {name: _RuleArg(arg) for name, arg in kwargs.items()}
        rule = rule_fn(*rule_args, **rule_kwargs)
        return _BoundIndexValueRule(rule, rule_args, rule_kwargs)

    def _install_bulk_rules(self) -> None:
        value_propagating_ops = OrderedSet(self._VALUE_PROPAGATING_OPS)
        torch._check(
            len(value_propagating_ops) == len(self._VALUE_PROPAGATING_OPS),
            lambda: "Duplicate value-propagating op classification",
        )
        for name in self._VALUE_PROPAGATING_OPS:
            setattr(self, name, self.value_propagating)

    def value_propagating(self, *args: Any, **kwargs: Any) -> _IndexValueRule:
        return _IndexValueRule(value_inputs=None)

    # op rules

    def bucketize(
        self,
        values: Any,
        boundaries: Any,
        boundary_indices: Any,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Any = None,
        sorter_indices: Any = None,
    ) -> _IndexValueRule:
        return _IndexValueRule(
            indexing_sinks=(
                boundaries,
                boundary_indices,
                sorter,
                sorter_indices,
            ),
            value_sinks=(values,),
        )

    def getitem(self, value: Any, index: Any) -> _IndexValueRule:
        return _IndexValueRule(value_inputs=(value,))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> _IndexValueRule:
        return _IndexValueRule(indexing_sinks=(expr, size))

    def device_assert_async(self, cond: Any, msg: str) -> _IndexValueRule:
        return _IndexValueRule(indexing_sinks=(cond,))

    def indirect_indexing(
        self, x: Any, size: sympy.Expr, check: bool = True, wrap_neg: bool = True
    ) -> _IndexValueRule:
        return _IndexValueRule(indexing_sinks=(x, size))

    def load(self, name: str, index: sympy.Expr) -> _IndexValueRule:
        return _IndexValueRule(indexing_sinks=(index,))

    def load_seed(self, name: str, offset: Any) -> _IndexValueRule:
        return _IndexValueRule(indexing_sinks=(offset,))

    def masked(self, mask: Any, body: Any, other: Any) -> _IndexValueRule:
        return _IndexValueRule(
            indexing_sinks=(mask,),
            value_inputs=(other,),
        )

    def output(self, *args: Any) -> _IndexValueRule:
        return _IndexValueRule(value_sinks=args)

    def partial_accumulate(
        self, name: str, reduction_type: Any, value: Any, extra_meta: dict[str, Any]
    ) -> _IndexValueRule:
        return _IndexValueRule(value_sinks=(value,))

    def placeholder(self, index: int) -> _IndexValueRule:
        return _IndexValueRule()

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: Any,
        value: Any,
    ) -> _IndexValueRule:
        return _IndexValueRule(value_sinks=(value,))

    def scan(
        self, dtypes: tuple[torch.dtype, ...], combine_fn: Any, values: Any
    ) -> _IndexValueRule:
        return _IndexValueRule(value_sinks=(values,))

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: Any,
        stable: bool,
        descending: bool,
    ) -> _IndexValueRule:
        return _IndexValueRule(value_sinks=(values,))

    def store(
        self, name: str, index: sympy.Expr, value: Any, mode: Any = None
    ) -> _IndexValueRule:
        return _IndexValueRule(
            indexing_sinks=(index,),
            value_sinks=(value,),
        )

    def store_reduction(
        self, name: str, index: sympy.Expr, value: Any
    ) -> _IndexValueRule:
        return self.store(name, index, value)

    # LoopBody pseudo call_module rules.

    def masked_subblock(self, mask: Any, other: Any) -> _IndexValueRule:
        return _IndexValueRule(
            indexing_sinks=(mask,),
            value_inputs=(other,),
        )

    def scan_subblock(
        self, dtypes: tuple[torch.dtype, ...], values: Any
    ) -> _IndexValueRule:
        return _IndexValueRule(value_sinks=(values,))

    def set_indirect(self, new_var: Any) -> _IndexValueRule:
        return _IndexValueRule(indexing_sinks=(new_var,))


_INDEX_VALUE_OPS = _IndexValueOpsHandler()


def _is_masked_subblock(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_module"
        and isinstance(node.target, str)
        and node.target.startswith("masked_subblock")
    )


def _map_rule_arg(arg: Any, fn: Any) -> None:
    def visit_node(node: torch.fx.Node) -> torch.fx.Node:
        fn(node)
        return node

    def visit(elem: Any) -> Any:
        if isinstance(elem, _RuleArg):
            map_arg(elem.value, visit_node)
        elif isinstance(elem, torch.fx.Node):
            fn(elem)
        return elem

    map_aggregate(arg, visit)


def _rewrite_rule_arg(arg: Any, fn: Any) -> None:
    def visit(elem: Any) -> Any:
        if isinstance(elem, _RuleArg):
            elem.value = fn(elem.value)
        return elem

    map_aggregate(arg, visit)


def _lint_loop_body_graph(graph: torch.fx.Graph) -> None:
    owning_module = graph.owning_module
    try:
        graph.owning_module = None
        graph.lint()
    finally:
        graph.owning_module = owning_module


def _compute_graph_uses(
    graph: torch.fx.Graph,
    *,
    output_is_indexing: bool = False,
    output_is_value: bool = True,
) -> tuple[OrderedSet[torch.fx.Node], OrderedSet[torch.fx.Node]]:
    indexing_sinks: OrderedSet[torch.fx.Node] = OrderedSet()
    value_sinks: OrderedSet[torch.fx.Node] = OrderedSet()

    def _add(target: OrderedSet[torch.fx.Node], arg: Any) -> None:
        _map_rule_arg(arg, target.add)

    for node in graph.nodes:
        if node.op == "output":
            if output_is_indexing:
                _add(indexing_sinks, node.args)
            if output_is_value:
                _add(value_sinks, node.args)
            continue

        node_rule = _INDEX_VALUE_OPS.rule_for_node(node)
        if node_rule is None:
            continue

        for arg in node_rule.rule.indexing_sinks:
            _add(indexing_sinks, arg)
        for arg in node_rule.rule.value_sinks:
            _add(value_sinks, arg)

    indexing_use = _mark_indexing_ancestors(indexing_sinks)
    value_use = _mark_value_ancestors(value_sinks, indexing_use)
    return indexing_use, value_use


def _graph_output_contexts(
    loop_body: LoopBody,
) -> dict[torch.fx.Graph, tuple[bool, bool]]:
    subblocks = {
        name: block.graph for name, block in getattr(loop_body, "subblocks", {}).items()
    }
    contexts = {loop_body.root_block.graph: (False, True)}
    worklist = [loop_body.root_block.graph]

    while worklist:
        graph = worklist.pop()
        output_is_indexing, output_is_value = contexts[graph]
        indexing_use, value_use = _compute_graph_uses(
            graph,
            output_is_indexing=output_is_indexing,
            output_is_value=output_is_value,
        )
        for node in graph.nodes:
            if not _is_masked_subblock(node):
                continue
            assert isinstance(node.target, str)
            subblock_graph = subblocks.get(node.target)
            if subblock_graph is None:
                continue

            child_is_indexing = node in indexing_use
            # Policy choice for this pass: if the same masked-subblock result
            # is both index-used and value-used, keep its body on the indexing
            # path instead of cloning the subblock for a separate value result.
            child_is_value = node in value_use and not child_is_indexing
            old_is_indexing, old_is_value = contexts.get(subblock_graph, (False, False))
            new_is_indexing = old_is_indexing or child_is_indexing
            new_context = (
                new_is_indexing,
                (old_is_value or child_is_value) and not new_is_indexing,
            )
            if new_context != (old_is_indexing, old_is_value):
                contexts[subblock_graph] = new_context
                worklist.append(subblock_graph)

    for graph in subblocks.values():
        contexts.setdefault(graph, (False, False))
    return contexts


def _mark_indexing_ancestors(
    starts: OrderedSet[torch.fx.Node],
) -> OrderedSet[torch.fx.Node]:
    """
    Walk backward from `starts`, accumulating ancestors. `load` is a barrier:
    its output value is determined by what is at the loaded address, not by
    the index expression, so usage does not propagate into the load's index.
    """
    marked: OrderedSet[torch.fx.Node] = OrderedSet()
    stack: list[torch.fx.Node] = list(starts)
    while stack:
        node = stack.pop()
        if node in marked:
            continue
        marked.add(node)

        def append_node(n: torch.fx.Node) -> torch.fx.Node:
            stack.append(n)
            return n

        target = node.target
        if target == "load":
            continue
        if _is_masked_subblock(node):
            node_rule = _INDEX_VALUE_OPS.rule_for_node(node)
            if node_rule is not None:
                value_inputs = node_rule.rule.value_inputs
                assert value_inputs is not None
                for arg in value_inputs:
                    _map_rule_arg(arg, append_node)
            continue

        map_arg(node.args, append_node)
        map_arg(node.kwargs, append_node)
    return marked


def _mark_value_ancestors(
    starts: OrderedSet[torch.fx.Node],
    indexing_use: OrderedSet[torch.fx.Node],
) -> OrderedSet[torch.fx.Node]:
    """
    Walk backward from value sinks through value-propagating ops only.

    Unknown targets stop value-use propagation, so upstream ``index_expr`` nodes
    keep their indexing-path behavior. Mixed masked-subblock outputs are treated
    as indexing-only, so value flow does not enter a masked subblock that is
    already on the indexing path.
    """
    marked: OrderedSet[torch.fx.Node] = OrderedSet()
    stack: list[torch.fx.Node] = list(starts)
    while stack:
        node = stack.pop()
        if node in marked:
            continue
        marked.add(node)

        def append_node(n: torch.fx.Node) -> torch.fx.Node:
            stack.append(n)
            return n

        target = node.target
        if target == "load":
            continue

        node_rule = _INDEX_VALUE_OPS.rule_for_node(node)
        if node_rule is None:
            continue
        if _is_masked_subblock(node) and node in indexing_use:
            continue
        if node_rule.rule.value_inputs is not None:
            for arg in node_rule.rule.value_inputs:
                _map_rule_arg(arg, append_node)
            continue

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
    index_name = get_index_node.args[0] if len(get_index_node.args) > 0 else None
    if not isinstance(index_name, str):
        return None
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
        the original ``index_expr``. A mixed masked-subblock output is treated
        as indexing-only by policy above, avoiding subblock cloning.

    This is a safety net for legacy lowering callsites that still emit
    ``index_expr`` for value uses. New lowerings should prefer to emit
    ``value_expr`` directly when their intent is a value computation.
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

    output_contexts = _graph_output_contexts(loop_body)
    graph_value_uses: dict[torch.fx.Graph, OrderedSet[torch.fx.Node]] = {}

    def rewrite_graph(graph: torch.fx.Graph) -> None:
        output_is_indexing, output_is_value = output_contexts[graph]
        indexing_use, value_use = _compute_graph_uses(
            graph,
            output_is_indexing=output_is_indexing,
            output_is_value=output_is_value,
        )
        rewritten_value_use = OrderedSet(value_use)
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
                    rewritten_value_use.add(clone)
                    value_clones[node] = clone
                return value_clones[node]

            node_rule = _INDEX_VALUE_OPS.rule_for_node(node)
            if node_rule is None:
                return node

            value_inputs = node_rule.rule.value_inputs
            if value_inputs is not None:
                if not value_inputs:
                    return node
                if node not in indexing_use:
                    rewrite_rule_args(node, node_rule, value_inputs)
                    return node

                if node not in value_clones:
                    with graph.inserting_before(anchor):
                        clone = graph.node_copy(node, lambda n: n)
                    clone.meta = node.meta.copy()
                    clone_rule = _INDEX_VALUE_OPS.rule_for_node(clone)
                    assert clone_rule is not None
                    clone_value_inputs = clone_rule.rule.value_inputs
                    assert clone_value_inputs is not None
                    rewrite_rule_args(clone, clone_rule, clone_value_inputs)
                    rewritten_value_use.add(clone)
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
                rewritten_value_use.add(clone)
                value_clones[node] = clone
            return value_clones[node]

        def rewrite_value_arg(arg: Any, anchor: torch.fx.Node) -> Any:
            return map_arg(arg, lambda n: value_version(n, anchor))

        def rewrite_rule_args(
            node: torch.fx.Node,
            node_rule: _BoundIndexValueRule,
            sinks: tuple[Any, ...],
        ) -> None:
            for arg in sinks:
                _rewrite_rule_arg(arg, lambda value: rewrite_value_arg(value, node))

            arg_prefix = node.args[:1] if node.op == "call_method" else ()
            node.args = (*arg_prefix, *(arg.value for arg in node_rule.args))
            node.kwargs = {name: arg.value for name, arg in node_rule.kwargs.items()}

        for node in graph.nodes:
            if node.op == "output":
                if output_is_value:
                    node.args = rewrite_value_arg(node.args, node)
                continue

            node_rule = _INDEX_VALUE_OPS.rule_for_node(node)
            if node_rule is None:
                continue
            if node_rule.rule.value_sinks:
                rewrite_rule_args(node, node_rule, node_rule.rule.value_sinks)
            if (
                _is_masked_subblock(node)
                and node in value_use
                and node not in indexing_use
            ):
                value_inputs = node_rule.rule.value_inputs
                assert value_inputs is not None
                rewrite_rule_args(node, node_rule, value_inputs)

        graph_value_uses[graph] = rewritten_value_use
        _lint_loop_body_graph(graph)

    for graph in graphs:
        rewrite_graph(graph)

    bound_vars = loop_body.bounds()
    bounds = bound_vars.get_bounds()
    for graph in graphs:
        value_use = graph_value_uses[graph]
        for node in graph.find_nodes(op="call_method", target="value_expr", sort=False):
            if node not in value_use:
                continue
            dtype = _compute_value_expr_dtype(
                loop_body, node, bounds, bound_vars.replacement_vals, value_use
            )
            if dtype is not None:
                args = list(node.args)
                args[2] = dtype
                node.args = tuple(args)
        _lint_loop_body_graph(graph)
