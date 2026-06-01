import math
from dataclasses import dataclass
from typing import Any, Optional

import sympy

import torch
from torch.fx.node import map_arg
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

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


@dataclass(frozen=True)
class _ValueUseRule:
    # None means all node inputs are value inputs, except indexing_inputs.
    value_inputs: tuple[Any, ...] | None = None
    value_sinks: tuple[Any, ...] = ()
    indexing_inputs: tuple[Any, ...] = ()


class _ValueUseRules:
    """
    Classify which inputs receive value-use demand.
    """

    # Singleton _ValueUseRules, because we meta program over a number of op rules.
    # Those are only defined after other inductor state has run.
    _instance: Optional["_ValueUseRules"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        for op in OP_NAMES:
            if not hasattr(self, op):
                setattr(self, op, self.default_rule)

        unimplemented_ops = OP_NAMES - OrderedSet(dir(self))
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented value-use rule for ops: {unimplemented_ops}",
        )
        self._initialized = True

    @staticmethod
    def default_rule(*args: Any, **kwargs: Any) -> _ValueUseRule:
        return _ValueUseRule()

    def load(self, name: str, index: sympy.Expr) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(), indexing_inputs=(index,))

    def load_seed(self, name: str, offset: int) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(), indexing_inputs=(offset,))

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: Any,
        mode: Any = None,
    ) -> _ValueUseRule:
        return _ValueUseRule(
            value_sinks=(value,),
            indexing_inputs=(index,),
        )

    def store_reduction(
        self, name: str, index: sympy.Expr, value: Any
    ) -> _ValueUseRule:
        return _ValueUseRule(
            value_sinks=(value,),
            indexing_inputs=(index,),
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: Any,
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(value,))

    def partial_accumulate(
        self,
        name: str,
        reduction_type: str,
        value: Any,
        extra_meta: dict[str, Any],
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(value,))

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[Any, ...],
        stable: bool,
        descending: bool,
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(values,))

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Any,
        values: tuple[Any, ...],
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(values,))

    def bucketize(
        self,
        values: Any,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: Any,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: Any | None = None,
    ) -> _ValueUseRule:
        return _ValueUseRule(
            value_inputs=(values,),
            indexing_inputs=(boundaries, boundary_indices, sorter, sorter_indices),
        )

    def indirect_indexing(
        self, x: Any, size: sympy.Expr, check: bool = True, wrap_neg: bool = True
    ) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(), indexing_inputs=(x, size))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(), indexing_inputs=(expr, size))

    def masked(self, mask: Any, body: Any, other: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def masked_subblock(self, mask: Any, other: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def scan_subblock(
        self, dtypes: tuple[torch.dtype, ...], values: tuple[Any, ...]
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(values,))

    def set_indirect(self, new_var: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(), indexing_inputs=(new_var,))

    def device_assert_async(self, cond: Any, msg: str) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(), indexing_inputs=(cond,))


def _rule_for_node(node: torch.fx.Node, rules: _ValueUseRules) -> _ValueUseRule:
    if node.op == "call_method" and isinstance(node.target, str):
        rule_fn = getattr(rules, node.target, None)
        if rule_fn is not None:
            return rule_fn(*node.args[1:], **node.kwargs)
    elif node.op == "call_module" and isinstance(node.target, str):
        if node.target.startswith("masked_subblock"):
            return rules.masked_subblock(*node.args, **node.kwargs)
        if node.target.startswith("scan"):
            return rules.scan_subblock(*node.args, **node.kwargs)
        if node.target.startswith("set_indirect"):
            return rules.set_indirect(*node.args, **node.kwargs)

    return _ValueUseRule()


def _collect_nodes(arg: Any) -> set[torch.fx.Node]:
    nodes: set[torch.fx.Node] = set()

    def add_node(node: torch.fx.Node) -> torch.fx.Node:
        nodes.add(node)
        return node

    map_arg(arg, add_node)
    return nodes


def _convert_value_use_index_exprs(loop_body: LoopBody) -> bool:
    """
    Walk backward from value sinks through value-propagating inputs.
    Returns True if any index_expr node was rewritten.
    """
    value_use: set[torch.fx.Node] = set()
    stack: list[tuple[torch.fx.Graph, torch.fx.Node]] = []
    rules = _ValueUseRules()
    root_graph = loop_body.root_block.graph
    subblocks = getattr(loop_body, "subblocks", {})
    indirect_vars = getattr(loop_body, "indirect_vars", ())
    graphs = [
        root_graph,
        *(block.graph for block in subblocks.values()),
    ]
    indirect_inputs: dict[sympy.Symbol, tuple[torch.fx.Graph, Any]] = {}
    graph_outputs: dict[torch.fx.Graph, Any] = {}
    node_rules: dict[torch.fx.Node, _ValueUseRule] = {}

    def _add_node(graph: torch.fx.Graph, node: torch.fx.Node) -> torch.fx.Node:
        stack.append((graph, node))
        return node

    def _add_arg(graph: torch.fx.Graph, arg: Any) -> None:
        map_arg(arg, lambda node: _add_node(graph, node))

    def _add_graph_output(graph: torch.fx.Graph) -> None:
        _add_arg(graph, graph_outputs[graph])

    has_index_expr = False
    for graph in graphs:
        for node in graph.nodes:
            if node.op == "output":
                graph_outputs[graph] = node.args
                continue
            has_index_expr = has_index_expr or node.target == "index_expr"
            if (
                node.op == "call_module"
                and isinstance(node.target, str)
                and node.target.startswith("set_indirect")
            ):
                idx = int(node.target[len("set_indirect") :])
                indirect_inputs[indirect_vars[idx]] = (graph, node.args)
            rule = _rule_for_node(node, rules)
            node_rules[node] = rule
            _add_arg(graph, rule.value_sinks)

    if not has_index_expr:
        return False

    changed = False
    _add_graph_output(root_graph)
    while stack:
        graph, node = stack.pop()
        if node in value_use:
            continue
        value_use.add(node)

        if node.target == "index_expr":
            node.target = "value_expr"
            changed = True

        if (
            node.op == "call_module"
            and isinstance(node.target, str)
            and node.target in subblocks
        ):
            _add_graph_output(subblocks[node.target].graph)

        if node.op == "call_module" and node.target == "get_index":
            expr = loop_body.indexing_exprs[node.args[0]]
            if isinstance(expr, sympy.Expr):
                for symbol in expr.free_symbols:
                    indirect_input = indirect_inputs.get(symbol)
                    if indirect_input is not None:
                        input_graph, input_arg = indirect_input
                        _add_arg(input_graph, input_arg)

        rule = node_rules.get(node)
        if rule is None:
            rule = _rule_for_node(node, rules)
            node_rules[node] = rule
        if rule.value_inputs is None:
            indexing_inputs = _collect_nodes(rule.indexing_inputs)
            for arg in (node.args, node.kwargs):
                for arg_node in _collect_nodes(arg):
                    if arg_node not in indexing_inputs:
                        _add_node(graph, arg_node)
        else:
            _add_arg(graph, rule.value_inputs)
        _add_arg(graph, rule.value_sinks)

    return changed


def convert_index_expr_to_value_expr(loop_body: LoopBody) -> None:
    """
    Rewrite ``index_expr`` nodes that participate in value computation to
    ``value_expr``, so codegen can honor the requested dtype instead of
    narrowing to the kernel's indexing dtype.

    Classification: walk backward from value sinks (store value, reduction,
    output). Indexing inputs to ops such as ``load``, ``store``,
    ``check_bounds``, and ``set_indirect`` do not receive value demand. Any
    ``index_expr`` reachable from a value sink is converted in-place to
    ``value_expr``. Indirect indexing is handled by following value-reachable
    ``get_index`` expressions through any referenced ``indirect*`` symbols back
    to the corresponding ``set_indirect*`` input.

    TODO: if mixed use (same index_expr used for both indexing and value)
    ever occurs in practice, the node should be cloned so the indexing path
    keeps the original index_expr. This hasn't been observed so far.
    """
    if _convert_value_use_index_exprs(loop_body):
        LoopBody.get_nodes.clear_cache(loop_body)
        LoopBody.bounds.clear_cache(loop_body)
