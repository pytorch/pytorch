import functools
import itertools
import logging
import math
from typing import Dict, Iterable, Union

import sympy

import torch
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.utils._sympy.value_ranges import ValueRangeAnalysis, ValueRanges
from .ir import FloorDiv, InterpreterShim, LoopBody, ModularIndexing
from .utils import sympy_subs, sympy_symbol
from .virtualized import V

log = logging.getLogger(__name__)


def dominated_nodes(
    initial_queue: Union[torch.fx.Node, Iterable[torch.fx.Node]], skip_filter=None
):
    """Returns the set of nodes whose values depend on those within initial_queue"""
    if isinstance(initial_queue, torch.fx.Node):
        initial_queue = [initial_queue]

    dominated_set = set(initial_queue)

    while initial_queue:
        node = initial_queue.pop()
        for user in node.users:
            if skip_filter and skip_filter(user):
                continue
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    return dominated_set


def val_expressable_in_32_bits(val):
    if hasattr(val, "is_Boolean") and val.is_Boolean:
        return True

    if isinstance(val, sympy.Expr):
        assert val.is_constant()
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

    raise Exception(f"Unexpected value {val}")


def range_expressable_in_32_bits(range):
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(
        range.upper
    )


def get_expr_range(expr, vars_ranges: dict):
    fs = list(expr.free_symbols)
    if len(fs) == 0:
        return ValueRanges(expr, expr)

    vars_ranges = vars_ranges.copy()

    def replace_symbols_for_deriv(expr):
        cnt = itertools.count()

        # for the purposes of finding local, minimum, maximum, assume smoothness
        def mod_indexing_rep(x, y, z):
            if z.is_constant():
                new_var = sympy_symbol("mod_index" + f"{next(cnt)}")
                # TODO: check if x / y has a range <= z and return x / y.
                if z > 0:
                    vars_ranges[new_var] = ValueRanges(0, z - 1)
                else:
                    vars_ranges[new_var] = ValueRanges(z + 1, 0)
                fs.append(new_var)
                return new_var

            # never really happens, we'll bail on optimizing
            return (x / y) % z

        def indexing_div_rep(x, y):
            return x / y

        return expr.replace(ModularIndexing, mod_indexing_rep).replace(
            FloorDiv, indexing_div_rep
        )

    monotonic_increasing = []
    monotonic_decreasing = []
    other_symbols = []

    expr_for_deriv = replace_symbols_for_deriv(expr)
    for symbol in fs:
        diff = sympy.diff(expr_for_deriv, symbol)
        if diff.is_positive:
            monotonic_increasing.append(symbol)
        elif diff.is_positive is False:  # can return None
            monotonic_decreasing.append(symbol)
        else:
            # If diff_free_symbols only one symbol and it is the same as symbol,
            # If this symbol's lower and upper bounds are the same, then it is constant.
            # Add it to monotonic_increasing or monotonic_decreasing is ok.
            diff_free_symbols = list(diff.free_symbols)
            if (
                len(diff_free_symbols) == 1
                and symbol in diff_free_symbols
                and symbol in vars_ranges
                and vars_ranges[symbol].lower == vars_ranges[symbol].upper
            ):
                monotonic_increasing.append(symbol)
            else:
                other_symbols.append(symbol)

    if not other_symbols:
        max_val = sympy_subs(
            expr_for_deriv,
            {
                k: (v.upper if k in monotonic_increasing else v.lower)
                for k, v in vars_ranges.items()
            },
        )
        min_val = sympy_subs(
            expr_for_deriv,
            {
                k: (v.lower if k in monotonic_increasing else v.upper)
                for k, v in vars_ranges.items()
            },
        )
        if free_symbols(min_val):
            min_val = -math.inf
        if free_symbols(max_val):
            max_val = math.inf
        return ValueRanges(min_val, max_val)
    else:
        # bail on optimizing, have not run into this yet
        return ValueRanges(-math.inf, math.inf)


class OptimizeIndexing:
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32. This is an important optimization for indexing
    kernels such as Upsample and Interpolate.
    """

    def __init__(
        self,
        loop_body: LoopBody,
        indices_ranges: Dict[sympy.Symbol, int],
        indexing_exprs: Dict[str, sympy.Expr],
    ):
        self.loop_body = loop_body
        self.indices_range = indices_ranges
        self.indexing_exprs = indexing_exprs
        self.replacement_vals = {}
        self.interp_env = {}
        self.submodules = self.swap_submodules(dict(loop_body.submodules))

        indirect_var_set = set(loop_body.indirect_vars)
        self.index_indirect_dependecies = {
            index: expr.free_symbols & indirect_var_set
            for index, expr in indexing_exprs.items()
        }
        self.all_graphs = [loop_body.root_block.graph] + [
            block.graph for block in loop_body.subblocks.values()
        ]

        for k, v in indices_ranges.items():
            if free_symbols(v):
                v = math.inf
            self.replace_indirect(k, ValueRanges(0, v))

        # avoid computing these values, pessimistically assume that they are unbounded
        self.tensor_values_set = dominated_nodes(
            [
                node
                for node in self.all_nodes
                if node.target in ["load", "reduction"]
                or "masked_subblock" in node.target
            ]
        )

    def run(self):
        """Compute Value Ranges and try reduce precision of 'to_dtype' nodes to int32 where possible"""

        int64_dtype_nodes = [
            node
            for node in self.all_nodes
            if (
                node.target == "to_dtype"
                and node.args[2] == torch.int64
                and node not in self.tensor_values_set
            )
        ]
        if not int64_dtype_nodes:
            return

        for node in self.tensor_values_set:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self.interp_env[node] = torch._inductor.optimize_indexing.ValueRanges(
                    -math.inf, math.inf
                )

        interpreter = InterpreterShim(self.loop_body.root_block.graph, self.submodules)
        interpreter.run(V.get_ops_handler(), initial_env=self.interp_env)

        # TODO - if dominated node of one to_dtype is not expressible in int32,
        # we should short circuit another to_dtype node if that node also dominates
        for node in int64_dtype_nodes:
            self.try_to_reduce_precision(node)

    def try_to_reduce_precision(self, node):
        # if a downstream use of a node explicitly converts to int32, or float16/float32/float64,
        # then it's precision is set for that chain of uses, and we don't need to consider those
        # dominated values
        def skip_filter(node):
            return node.target == "to_dtype" and node.args[2] in (
                torch.int32,
                torch.float32,
                torch.float64,
            )

        # TODO - there are dominated uses whose dtype does not depend on whether
        # we reduce the precision here, e.g. add(int64, int64) one of the args can be reduced to
        # int32 without changing the output precision of the node. this case hasn't shown up
        for dominated in dominated_nodes(node, skip_filter):
            if dominated.target in ["store", "output"]:
                continue

            if "set_indirect" in dominated.target:
                idx = int(dominated.target[len("set_indirect") :])
                indirect_var = self.loop_body.indirect_vars[idx]

                for index, indirect_vals in self.index_indirect_dependecies.items():
                    if indirect_var in indirect_vals:
                        index_val = self.replacement_vals[index]

                        if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                            return

                        # all indices are integers, so make sure that we
                        # use the bounds of integers instead of floats.
                        # TODO - not sure if we should be doing int/float casts while tracing,
                        # might interfere with sympy.

                        index_val_int = ValueRanges(
                            int(index_val.lower), int(index_val.upper)
                        )
                        if not range_expressable_in_32_bits(index_val_int):
                            return

            if not range_expressable_in_32_bits(self.interp_env[dominated]):
                return

        args = list(node.args)
        args[2] = torch.int32
        node.args = tuple(args)

    @property
    def all_nodes(self):
        for graph in self.all_graphs:
            yield from graph.nodes

    def swap_submodules(self, submodules):
        keys = list(submodules.keys())
        for key in keys:
            if key == "get_index":
                submodules[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                submodules[key] = functools.partial(
                    self.masked_subblock, subblock, self.interp_env
                )
            else:
                assert "set_indirect" in key
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = functools.partial(self.set_indirect, var)
                submodules[key] = indirect

        return submodules

    def masked_subblock(self, subblock, env, mask, value):
        interp = InterpreterShim(subblock.graph, self.submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        assert len(output) == 1
        # dont bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, var, new_var):
        self.replace_indirect(var, new_var)
        return new_var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        assert isinstance(new, ValueRanges)
        self.replacement_vals[old] = new

    def get_index(self, name):
        if name in self.replacement_vals:
            return self.replacement_vals[name]

        out = self._get_index_impl(name)
        self.replacement_vals[name] = out
        return out

    def _get_index_impl(self, name):
        expr = self.indexing_exprs[name]
        if expr in self.replacement_vals:
            return self.replacement_vals[expr]
        return get_expr_range(expr, self.replacement_vals)


def indexing_dtype_strength_reduction(loop_body: LoopBody):
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
    indices = dict(loop_body.var_ranges)
    indexing = dict(loop_body.indexing_exprs)
    with V.set_ops_handler(ValueRangeAnalysis()):
        OptimizeIndexing(loop_body, indices, indexing).run()
