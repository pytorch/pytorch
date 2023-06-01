from functools import partial
from typing import Dict, Optional

import torch
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
from .ir import InterpreterShim, LoopBody
from .utils import cache_on_self, dominated_nodes
from .virtualized import V


def get_expr_range(expr, vars_ranges: dict):
    free_symbols = list(expr.free_symbols)
    if len(free_symbols) == 0:
        return ValueRanges(expr, expr)

    def replace_symbols_for_deriv(expr):
        # for the purposes of finding local, minimum, maximum, assume smoothness
        def mod_indexing_rep(x, y, z):
            if z.is_constant():
                return x / y

            # never really happens, we'll bail on optimizing
            return (x / y) % z

        def indexing_div_rep(x, y):
            return x / y

        return expr.replace(ModularIndexing, mod_indexing_rep).replace(
            FloorDiv, indexing_div_rep
        )

    symbols = expr.free_symbols
    monotonic_increasing = []
    monotonic_decreasing = []
    other_symbols = []

    expr_for_deriv = replace_symbols_for_deriv(expr)
    for symbol in symbols:
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
                and vars_ranges[symbol].lower == vars_ranges[symbol].upper
            ):
                monotonic_increasing.append(symbol)
            else:
                other_symbols.append(symbol)

    if not other_symbols:
        max_val = sympy_subs(
            expr,
            {
                k: (v.upper if k in monotonic_increasing else v.lower)
                for k, v in vars_ranges.items()
            },
        )
        min_val = sympy_subs(
            expr,
            {
                k: (v.lower if k in monotonic_increasing else v.upper)
                for k, v in vars_ranges.items()
            },
        )
        return ValueRanges(min_val, max_val)
    else:
        # bail on optimizing, have not run into this yet
        return ValueRanges(-math.inf, math.inf)


class BoundVars:
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable
    """

    def __init__(self, loop_body: LoopBody):
        self.loop_body = loop_body
        self.replacement_vals = {
            k: ValueRanges(0, v) for k, v in loop_body.var_ranges.items()
        }
        # avoid computing these values, pessimistically assume that they are unbounded
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ["load", "reduction"] or "masked_subblock" in node.target
        )
        # To access this variable call `get_bounds()`
        self._bounds: Optional[Dict[torch.fx.Node, ValueRanges]] = {}

    @cache_on_self
    def get_bounds(self):
        submodules = self.swap_submodules(self.loop_body.submodules)

        # Initialize the environment with the unbounded variables
        for node in self.unbounded_vars:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges.unknown()

        with V.set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(self, submodules):
        result = {}
        for key in submodules.keys():
            if key == "get_index":
                result[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                # The result within the lambda will reference to the final
                # set of modules at the end of the for-loop as it stores a reference to it
                result[key] = lambda mask, value: self.masked_subblock(
                    subblock, self._bounds, mask, value, result
                )
            else:
                assert "set_indirect" in key
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = partial(self.set_indirect, var)
                result[key] = indirect

        return result

    def masked_subblock(self, subblock, env, mask, value, submodules):
        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        assert len(output) == 1
        # dont bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, old, new):
        assert isinstance(new, ValueRanges)
        self.replacement_vals[old] = new
        return new

    def get_index(self, name):
        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is not None:
            return bound

        bound = bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound
