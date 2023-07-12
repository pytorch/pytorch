import math
from functools import partial
from typing import Dict, Optional

import torch
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
from .ir import InterpreterShim, LoopBody
from .utils import cache_on_self, dominated_nodes
from .virtualized import V


class BoundVars:
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable
    """

    def __init__(self, loop_body: LoopBody):
        self.loop_body = loop_body
        self.replacement_vals = {
            k: ValueRanges(0, v if not free_symbols(v) else math.inf)
            for k, v in loop_body.var_ranges.items()
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
        if bound is None:
            bound = bound_sympy(expr, self.replacement_vals)
        # The following assertion is true at the time of this writing
        # We don't assert is as to not execute bound_sympy when bound is not None
        # assert bound is None or bound == bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound
