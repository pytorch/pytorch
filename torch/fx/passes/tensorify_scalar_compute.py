# TODO: This mucks a bit around with Dynamo data structures, maybe put in
# Dynamo

import logging
import operator
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch.fx._utils import get_node_context, lazy_format_graph_code
from torch.fx.experimental.sym_node import SymNode
from torch.fx.graph_module import GraphModule
from torch._dynamo.variables.builder import (
    BackwardStateGraphArg
)

# Import sympy and ShapeEnv during TYPE_CHECKING since importing sympy is slow
if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")

def tensorify_scalar_compute(gm: GraphModule, shape_env: ShapeEnv) -> None:
    import sympy
    from torch.fx.experimental.sym_node import SymTypes
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, free_symbols

    graph = gm.graph

    # Dynamo already enforces that there are no float input/outputs to a
    # graph.  However, it generate a graph that will take a float input
    # denoting a Tensor and immediately call item() on it, and you'll still
    # end up with a pile of float intermediate compute.  The goal of this pass
    # is to eliminate this scalar compute, turning it into Tensor compute.
    #
    # The general shape of this transformation is to push item() calls as far
    # down as possible.
    #
    # For example, convert:
    #
    #   a: Sym(zf0) = x.item()
    #   b: Sym(zf0 + 3) = a + 3
    #
    # into
    #
    #   a: f64[] = x + 3
    #   b: Sym(zf0 + 3) = a.item()
    #
    # And we can eliminate it entirely if we hit a Tensor operation for which
    # we have a Tensor overload for the Scalar overload:
    #
    #   a: Sym(zf0) = x.item()
    #   b: f32[50] = tensor + a
    #
    # into
    #
    #   b: f32[50] = torch.add(tensor, x)
    #
    # (I've omitted type promotion from the translations here for clarity.)
    #
    # There are a few parts to this transformation:
    #
    #   1. We need a mapping of Python float compute to their Tensor
    #      equivalents (float -> float to Tensor -> Tensor)
    #
    #   2. We need a mapping of Python float-Tensor operations to
    #      Tensor-Tensor equivalents (e.g., (Tensor, float) -> Tensor
    #      to (Tensor, Tensor) -> Tensor.  For simplicity, these mappings
    #      will be defined only for ATen operators, but in principle you could
    #      also expand it to cover pre-dispatch IR or regular Dynamo IR.
    #
    # There is some interaction with SymInt.  Specifically, what should we do
    # if a float turns into an int?  Maybe not, if we are going to eventually
    # going to use it in a sizey way.  But if it goes to a Tensor, maybe we
    # should!
    #
    # This suggests that we want some sort of analysis



    return

    #
    #
    # The zf0 symbols need some special handling in produce_guards, as there
    # is no longer an input that directly generates them.  There's also a type
    # mismatch between what Dynamo sees as the user inputs, and what the FX
    # graph seeas as inputs.  TODO: resolve this.

    for node in graph.nodes:
        if node.op == "placeholder":
            zf0 = node.meta["grapharg"].example
            if isinstance(zf0, torch.SymFloat) and isinstance(zf0.node.expr, sympy.Symbol):
                # Binding site for SymFloat, swizzle it
                with graph.inserting_before(node):
                    graph.placeholder(node.name)


    used_symbols = set()
    for node in graph.nodes:
        binds_symbol = placeholder_binds_symbol(node) is not None
        if not binds_symbol:
            # Register the free symbols as uses
            arg = node.meta.get("grapharg")
            if isinstance(arg, BackwardStateGraphArg):  # TODO: blegh
                continue
            fake = (
                arg.fake_tensor if arg.fake_tensor is not None else arg.example
            )
            used_symbols |= free_symbols(fake)

    # Force specialization
    for s in used_symbols:
        # NB: get here because unbacked symbols may show up here, we don't
        # care about those
        if type(val := shape_env.var_to_val.get(s)) is float:
            log.info("forcing specialization %s = %s", s, val)
            shape_env.evaluate_expr(val)

