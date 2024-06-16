# mypy: allow-untyped-defs
import logging
import operator
from functools import partial
from typing import Any, Callable, Dict

from sympy import Expr

import torch
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
from .ir import InterpreterShim, LoopBody, LoopBodyBlock
from .utils import cache_on_self, dominated_nodes
from .virtualized import V


log = logging.getLogger(__name__)


class BoundVars:
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable

    Note. A current limitation of this analysis is that it just works on a per-loop basis.
    We should be able to propagate the bounds between across the whole graph. This may benefit
    the case a bounded variable is returned by a kernel and fed into another.
    """

    def __init__(self, loop_body: LoopBody) -> None:
        def upper_bound(v):
            return bound_sympy(v).upper if isinstance(v, Expr) else v

        self.loop_body = loop_body
        self.replacement_vals = {
            k: ValueRanges[Expr](0, upper_bound(v) - 1)
            for k, v in loop_body.var_ranges.items()
        }
        # avoid computing these values, pessimistically assume that they are unbounded
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ["load", "reduction", operator.getitem]
            or "masked_subblock" in node.target
        )
        # To access this variable call `get_bounds()`
        self._bounds: Dict[torch.fx.Node, ValueRanges[Expr]] = {}

    @cache_on_self
    def get_bounds(self) -> Dict[torch.fx.Node, ValueRanges[Expr]]:
        submodules = self.swap_submodules(self.loop_body.submodules)

        # Initialize the environment with the unbounded variables
        for node in self.unbounded_vars:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if not isinstance(node.target, str) or (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges[Expr].unknown()

        with V.set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            log.debug("get_bounds:\n%s", self.loop_body.root_block.graph)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(
        self, submodules: Dict[str, Callable[..., Any]]
    ) -> Dict[str, Callable[..., ValueRanges[Expr]]]:
        result: Dict[str, Callable[..., ValueRanges[Expr]]] = {}
        for key in submodules.keys():
            if key == "get_index":
                result[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                # The result within the lambda will reference to the final
                # set of modules at the end of the for-loop as it stores a reference to it

                # bind subblock in a function because python lambdas close over by reference
                # moving the lambda out of make_fn would close over the reference to subblock,
                # so all lambdas would have the same subblock reference that is the final
                # subblock in the loop
                def make_fn(subblock):
                    return lambda mask, value: self.masked_subblock(
                        subblock, self._bounds, mask, value, result
                    )

                result[key] = make_fn(subblock)
            elif "set_indirect" in key:
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = partial(self.set_indirect, var)
                result[key] = indirect
            else:
                assert "scan" in key
                result[key] = submodules[key]

        return result

    def masked_subblock(
        self,
        subblock: LoopBodyBlock,
        env: Dict[torch.fx.Node, ValueRanges[Expr]],
        mask: Any,
        value: Any,
        submodules: Dict[str, Callable[..., Any]],
    ) -> ValueRanges[Expr]:
        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        assert len(output) == 1
        # dont bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges[Expr]) -> ValueRanges[Expr]:
        assert isinstance(new, ValueRanges)
        self.replacement_vals[old] = new
        return new

    def get_index(self, name: Expr) -> ValueRanges[Expr]:
        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is None:
            bound = bound_sympy(expr, self.replacement_vals)
        # The following assertion is true at the time of this writing
        # We don't assert is as to not execute bound_sympy when bound is not None
        # assert bound is None or bound == bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound
