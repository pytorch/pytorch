import math
import operator
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.fx.node import map_aggregate, map_arg
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


def _dominated_uses_fit_in_32_bits(
    node: torch.fx.Node,
    bounds: dict[Any, ValueRanges[Any]],
    indirect_vars: list[Any],
    indices: dict[Any, sympy.Expr],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
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


@dataclass(frozen=True)
class _IndexValueRule:
    # Inputs in the LoopBody FX node that act as terminal sinks.
    indexing_sinks: tuple[Any, ...] = ()
    value_sinks: tuple[Any, ...] = ()
    # Inputs that receive value use from this op's result. None means all inputs.
    value_inputs: tuple[Any, ...] | None = ()


_INDEXING_DEMAND = 1
_VALUE_DEMAND = 2


@dataclass(frozen=True)
class _ArgRef:
    key: int | str

    def get(self, node: torch.fx.Node) -> Any:
        if isinstance(self.key, int):
            return node.args[self.key]
        return node.kwargs[self.key]

    def update(self, node: torch.fx.Node, fn: Any) -> None:
        value = fn(self.get(node))
        if isinstance(self.key, int):
            args = list(node.args)
            args[self.key] = value
            node.args = tuple(args)
        else:
            kwargs = dict(node.kwargs)
            kwargs[self.key] = value
            node.kwargs = kwargs


class _IndexValueOpsHandler:
    """
    Classify each OpsHandler op for index/value-use analysis.
    """

    _instance: Any = None

    _EXTRA_VALUE_PROPAGATING_OPS = (
        "ceil_to_int",
        "constant",
        "div_rn",
        "dot",
        "floor",
        "floor_to_int",
        "floordiv",
        "fmod",
        "frexp",
        "halide_clamp",
        "identity",
        "index_expr",
        "inline_asm_elementwise",
        "int_truediv",
        "lshift",
        "mod",
        "mul",
        "pow",
        "rand",
        "rand_eager",
        "randint64",
        "randn",
        "round",
        "round_to_int",
        "rshift",
        "to_dtype",
        "to_dtype_bitcast",
        "truediv",
        "trunc",
        "trunc_to_int",
        "truncdiv",
        "value_expr",
        "where",
    )

    def __new__(cls) -> "_IndexValueOpsHandler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        registered_value_propagating_ops = self._registered_value_propagating_ops()
        self._value_propagating_ops = registered_value_propagating_ops | OrderedSet(
            self._EXTRA_VALUE_PROPAGATING_OPS
        )
        self._install_bulk_rules()
        unknown_ops = self._value_propagating_ops - OP_NAMES
        torch._check(
            len(unknown_ops) == 0,
            lambda: f"Value/index rules for unknown ops: {unknown_ops}",
        )
        duplicate_value_propagating_ops = len(self._value_propagating_ops) != len(
            registered_value_propagating_ops
        ) + len(self._EXTRA_VALUE_PROPAGATING_OPS)
        torch._check(
            not duplicate_value_propagating_ops,
            lambda: "Duplicate value-propagating op classification",
        )
        unimplemented_ops = OP_NAMES - OrderedSet(dir(self))
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented value/index rule for ops: {unimplemented_ops}",
        )
        self._initialized = True

    @classmethod
    def _registered_value_propagating_ops(cls) -> OrderedSet[str]:
        from .utils import boolean_ops, op_dtype_propagation_rules

        if not op_dtype_propagation_rules:
            from . import lowering  # noqa: F401

        from .codegen.common import pointwise_overrides_data

        return (
            OrderedSet(op_dtype_propagation_rules)
            | OrderedSet(pointwise_overrides_data)
            | OrderedSet(boolean_ops())
        ) & OrderedSet(OP_NAMES)

    def rule_for_node(self, node: torch.fx.Node) -> _IndexValueRule | None:
        if node.op == "output":
            return self._rule_for_node(self.output, node)
        if node.op == "call_function" and node.target is operator.getitem:
            return self._rule_for_node(self.getitem, node)
        if not isinstance(node.target, str):
            return None

        target = node.target
        if node.op == "call_module":
            if target.startswith("set_"):
                return self._rule_for_node(self.set_indirect, node)
            if target.startswith("masked_subblock"):
                return self._rule_for_node(self.masked_subblock, node)
            if target.startswith("scan"):
                return self._rule_for_node(self.scan_subblock, node)
            return None

        if target not in OP_NAMES:
            return None

        return self._rule_for_node(
            getattr(self, target),
            node,
            arg_start=1 if node.op == "call_method" else 0,
        )

    def _rule_for_node(
        self,
        rule_fn: Any,
        node: torch.fx.Node,
        *,
        arg_start: int = 0,
    ) -> _IndexValueRule:
        args = tuple(_ArgRef(i) for i in range(arg_start, len(node.args)))
        kwargs = {name: _ArgRef(name) for name in node.kwargs}
        return rule_fn(*args, **kwargs)

    def _install_bulk_rules(self) -> None:
        existing_ops = OrderedSet(
            name for name in self._value_propagating_ops if hasattr(self, name)
        )
        torch._check(
            len(existing_ops) == 0,
            lambda: f"Ops have both value-propagating and explicit rules: {existing_ops}",
        )
        for name in self._value_propagating_ops:
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


def _is_masked_subblock(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_module"
        and isinstance(node.target, str)
        and node.target.startswith("masked_subblock")
    )


def _map_rule_arg(node: torch.fx.Node, arg: Any, fn: Any) -> None:
    def visit_node(arg_node: torch.fx.Node) -> torch.fx.Node:
        fn(arg_node)
        return arg_node

    def visit(elem: Any) -> Any:
        if isinstance(elem, _ArgRef):
            map_arg(elem.get(node), visit_node)
        elif isinstance(elem, torch.fx.Node):
            fn(elem)
        return elem

    map_aggregate(arg, visit)


def _rewrite_rule_arg(node: torch.fx.Node, arg: Any, fn: Any) -> None:
    def visit(elem: Any) -> Any:
        if isinstance(elem, _ArgRef):
            elem.update(node, fn)
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
    demands: dict[torch.fx.Node, int] = {}
    handler = _IndexValueOpsHandler()

    def mark(node: torch.fx.Node, demand: int) -> None:
        demands[node] = demands.get(node, 0) | demand

    def mark_rule_arg(node: torch.fx.Node, arg: Any, demand: int) -> None:
        _map_rule_arg(node, arg, lambda n: mark(n, demand))

    def mark_args(node: torch.fx.Node, demand: int) -> None:
        def mark_arg(arg_node: torch.fx.Node) -> torch.fx.Node:
            mark(arg_node, demand)
            return arg_node

        map_arg(node.args, mark_arg)
        map_arg(node.kwargs, mark_arg)

    # FX graph order is definition-before-use, so reverse order lets each node
    # see all downstream demands before propagating to its inputs.
    for node in reversed(tuple(graph.nodes)):
        if node.op == "output":
            if output_is_indexing:
                mark_rule_arg(node, node.args, _INDEXING_DEMAND)
            if output_is_value:
                mark_rule_arg(node, node.args, _VALUE_DEMAND)
            continue

        rule = handler.rule_for_node(node)
        if rule is not None:
            for arg in rule.indexing_sinks:
                mark_rule_arg(node, arg, _INDEXING_DEMAND)
            for arg in rule.value_sinks:
                mark_rule_arg(node, arg, _VALUE_DEMAND)

        demand = demands.get(node, 0)
        if demand == 0 or node.op == "placeholder":
            continue

        if demand & _INDEXING_DEMAND and node.target != "load":
            if _is_masked_subblock(node):
                if rule is not None:
                    value_inputs = rule.value_inputs
                    assert value_inputs is not None
                    for arg in value_inputs:
                        mark_rule_arg(node, arg, _INDEXING_DEMAND)
            else:
                mark_args(node, _INDEXING_DEMAND)

        if demand & _VALUE_DEMAND and node.target != "load":
            if rule is None:
                continue
            if _is_masked_subblock(node) and demand & _INDEXING_DEMAND:
                continue
            if rule.value_inputs is not None:
                for arg in rule.value_inputs:
                    mark_rule_arg(node, arg, _VALUE_DEMAND)
            else:
                mark_args(node, _VALUE_DEMAND)

    return (
        OrderedSet(n for n in graph.nodes if demands.get(n, 0) & _INDEXING_DEMAND),
        OrderedSet(n for n in graph.nodes if demands.get(n, 0) & _VALUE_DEMAND),
    )


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

    def rewrite_graph(graph: torch.fx.Graph) -> None:
        output_is_indexing, output_is_value = output_contexts[graph]
        indexing_use, value_use = _compute_graph_uses(
            graph,
            output_is_indexing=output_is_indexing,
            output_is_value=output_is_value,
        )
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

            rule = _IndexValueOpsHandler().rule_for_node(node)
            if rule is None:
                return node

            value_inputs = rule.value_inputs
            if value_inputs is not None:
                if not value_inputs:
                    return node
                if node not in indexing_use:
                    rewrite_rule_args(node, value_inputs)
                    return node

                if node not in value_clones:
                    with graph.inserting_before(anchor):
                        clone = graph.node_copy(node, lambda n: n)
                    clone.meta = node.meta.copy()
                    clone_rule = _IndexValueOpsHandler().rule_for_node(clone)
                    assert clone_rule is not None
                    clone_value_inputs = clone_rule.value_inputs
                    assert clone_value_inputs is not None
                    rewrite_rule_args(clone, clone_value_inputs)
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

        def rewrite_rule_args(node: torch.fx.Node, sinks: tuple[Any, ...]) -> None:
            for arg in sinks:
                _rewrite_rule_arg(
                    node, arg, lambda value: rewrite_value_arg(value, node)
                )

        for node in graph.nodes:
            if node.op == "output":
                if output_is_value:
                    node.args = rewrite_value_arg(node.args, node)
                continue

            rule = _IndexValueOpsHandler().rule_for_node(node)
            if rule is None:
                continue
            if rule.value_sinks:
                rewrite_rule_args(node, rule.value_sinks)
            if (
                _is_masked_subblock(node)
                and node in value_use
                and node not in indexing_use
            ):
                value_inputs = rule.value_inputs
                assert value_inputs is not None
                rewrite_rule_args(node, value_inputs)

        _lint_loop_body_graph(graph)

    for graph in graphs:
        rewrite_graph(graph)

    LoopBody.get_nodes.clear_cache(loop_body)
    LoopBody.bounds.clear_cache(loop_body)
