import math
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.fx.node import map_arg
from torch.utils._ordered_set import OrderedSet
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


@dataclass(frozen=True)
class _ValueUseRule:
    # These fields are op arguments, so Any covers FX nodes, SymPy exprs,
    # scalars, and nested tuples/lists. value_sinks seed the backward walk
    # unconditionally. value_inputs propagate value demand only when the op's
    # result is already value-reachable. indexing_inputs block propagation
    # because they only affect addresses, bounds, masks, or other indexing-only
    # state.
    value_inputs: tuple[Any, ...] = ()
    value_sinks: tuple[Any, ...] = ()
    indexing_inputs: tuple[Any, ...] = ()


def _collect_fx_nodes(arg: Any) -> OrderedSet[torch.fx.Node]:
    nodes: OrderedSet[torch.fx.Node] = OrderedSet()

    def add_node(node: torch.fx.Node) -> torch.fx.Node:
        nodes.add(node)
        return node

    map_arg(arg, add_node)
    return nodes


def _collect_input_nodes(node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
    inputs = OrderedSet(node.all_input_nodes)
    if (
        node.op == "call_method"
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        inputs.discard(node.args[0])
    return inputs


class _ValueUseRules:
    @staticmethod
    def default_rule(*args: Any, **kwargs: Any) -> _ValueUseRule | None:
        return None

    def load(self, name: str, index: sympy.Expr) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(index,))

    def load_seed(self, name: str, offset: int) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(offset,))

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
        combine_fn_or_values: Any,
        values: tuple[Any, ...] | None = None,
    ) -> _ValueUseRule:
        if values is None:
            values = combine_fn_or_values
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
        return _ValueUseRule(indexing_inputs=(x, size))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(expr, size))

    def masked(self, mask: Any, body: Any, other: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def masked_subblock(self, mask: Any, other: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def set_indirect(self, new_var: Any) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(new_var,))

    def device_assert_async(self, cond: Any, msg: str) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(cond,))


class _ValueUseAnalysis:
    def __init__(self, loop_body: LoopBody) -> None:
        self.loop_body = loop_body
        self.rules = _ValueUseRules()
        self.root_graph = loop_body.root_block.graph
        self.subblocks = getattr(loop_body, "subblocks", {})
        self.indirect_vars = getattr(loop_body, "indirect_vars", ())
        self.graphs = [
            self.root_graph,
            *(block.graph for block in self.subblocks.values()),
        ]
        self.install_call_module_rules()

        self.value_reachable: OrderedSet[torch.fx.Node] = OrderedSet()
        self.worklist: list[tuple[torch.fx.Graph, torch.fx.Node]] = []
        self.indirect_inputs: dict[sympy.Symbol, tuple[torch.fx.Graph, Any]] = {}

    def run(self) -> bool:
        if not self.has_index_expr():
            return False

        self.seed_value_reachable_nodes()
        return self.propagate_value_reachability()

    def has_index_expr(self) -> bool:
        return any(
            graph.find_nodes(op="call_method", target="index_expr", sort=False)
            for graph in self.graphs
        )

    def install_call_module_rules(self) -> None:
        for graph in self.graphs:
            for node in graph.find_nodes(op="call_module", sort=False):
                if not isinstance(node.target, str):
                    continue
                if node.target in self.subblocks:
                    setattr(self.rules, node.target, self.rules.masked_subblock)
                elif node.target.startswith("scan"):
                    setattr(self.rules, node.target, self.rules.scan)
                elif node.target.startswith("set_indirect"):
                    setattr(self.rules, node.target, self.rules.set_indirect)

    def seed_value_reachable_nodes(self) -> None:
        self._enqueue_graph_output(self.root_graph)
        for graph in self.graphs:
            for node in graph.nodes:
                if node.op == "output":
                    continue
                if (
                    node.op == "call_module"
                    and isinstance(node.target, str)
                    and node.target.startswith("set_indirect")
                ):
                    idx = int(node.target[len("set_indirect") :])
                    self.indirect_inputs[self.indirect_vars[idx]] = (graph, node.args)
                for sink_node in self.value_sink_nodes(node):
                    self.worklist.append((graph, sink_node))

    def propagate_value_reachability(self) -> bool:
        changed = False
        while self.worklist:
            graph, node = self.worklist.pop()
            if node in self.value_reachable:
                continue
            self.value_reachable.add(node)

            if node.target == "index_expr":
                node.target = "value_expr"
                changed = True

            if (
                node.op == "call_module"
                and isinstance(node.target, str)
                and node.target in self.subblocks
            ):
                self._enqueue_graph_output(self.subblocks[node.target].graph)

            if node.op == "call_module" and node.target == "get_index":
                expr = self.loop_body.indexing_exprs[node.args[0]]
                if isinstance(expr, sympy.Expr):
                    for symbol in expr.free_symbols:
                        indirect_input = self.indirect_inputs.get(symbol)
                        if indirect_input is not None:
                            ig, ia = indirect_input
                            map_arg(ia, lambda n: self.worklist.append((ig, n)))

            for input_node in self.value_input_nodes(node):
                self.worklist.append((graph, input_node))

        return changed

    def rule_for_node(self, node: torch.fx.Node) -> _ValueUseRule | None:
        if node.op == "call_method" and isinstance(node.target, str):
            rule_fn = getattr(self.rules, node.target, self.rules.default_rule)
            return rule_fn(*node.args[1:], **node.kwargs)
        if node.op == "call_module" and isinstance(node.target, str):
            rule_fn = getattr(self.rules, node.target, self.rules.default_rule)
            return rule_fn(*node.args, **node.kwargs)
        return None

    def value_input_nodes(self, node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
        rule = self.rule_for_node(node)
        if rule is None:
            return _collect_input_nodes(node)
        return _collect_fx_nodes(rule.value_inputs) | _collect_fx_nodes(
            rule.value_sinks
        )

    def value_sink_nodes(self, node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
        rule = self.rule_for_node(node)
        return _collect_fx_nodes(rule.value_sinks) if rule else OrderedSet()

    def _enqueue_graph_output(self, graph: torch.fx.Graph) -> None:
        output_nodes = graph.find_nodes(op="output", sort=False)
        assert len(output_nodes) == 1
        map_arg(output_nodes[0].args, lambda n: self.worklist.append((graph, n)))


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

    Mixed-use policy: if the same ``index_expr`` feeds both an indexing use and
    a value use, rewrite it in-place to ``value_expr``. This is conservative and
    correct because the indexing path may compute at a wider dtype than needed,
    but the value path cannot lose precision. Cloning would preserve the
    narrower indexing path, but the mixed-use case is rare and the simpler
    in-place rewrite avoids extra CSE/register pressure.
    """
    if _ValueUseAnalysis(loop_body).run():
        LoopBody.get_nodes.clear_cache(loop_body)
        LoopBody.bounds.clear_cache(loop_body)
