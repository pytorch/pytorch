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

def tensorify_float_inputs(gm: GraphModule, shape_env: ShapeEnv) -> None:
    import sympy
    from torch.fx.experimental.sym_node import SymTypes
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, free_symbols

    graph = gm.graph

    return

    # We proceed by a sequence of local transforms.  Intuitively, we want to
    # mutate:
    #
    #   %a: SymFloat(zf0) = placeholder[target=x]
    #
    # into
    #
    #   %na: f64[] = placeholder[target=x]
    #   %a: SymFloat(...) = call_method[target=item](args = (%na,))
    #
    # The big hazard of this transformation is that, done naively, the
    # previously backed SymFloat become unbacked SymFloat, and then if you
    # repropagate through them you can end up with guard on data dependent
    # shape error.  Our plan is to have a special item() HOP, which produces
    # the original zf0 backed SymFloat when propagated.
    #
    # Note that our hope is that for the most part, SymFloats never appear in
    # guards, because you propagate them along until they get used as a Scalar
    # argument that doesn't guard on them.  However, we need to ensure
    # compilation doesn't fail even if you do use the SymFloat in a guard-ey
    # way, and so this tracking is necessary.
    #
    # Originally, I wanted to just force specialization when a float was used
    # in a "sizey" way to avoid having to deal with the complication here.
    # Unfortunately, it's not easy to tell if the unbacked SymFloat is going
    # to trigger more guards.  For example, if we have:
    #
    #   def f(y: f32[30], f0: f64[]):
    #       zuf0: SymFloat(zuf0) = f0.item()
    #       x: SymInt(floor(zuf0)) = floor(zuf0)
    #       return y.select(0, x)
    #
    # zuf0 never shows up inside of a Tensor size (it's solely being used to
    # decide where to index into y), but it will participate in guards on
    # subsequent meta propagation because whenever we run the meta for select
    # we must test if floor(zuf0) <= 30.  I cannot think of a good way to say
    # "ah yes, zuf0 should be specialized here", but conversely, to NOT
    # specialize it if we had torch.add(y, x) instead.  So we're just going to
    # do the complicated thing.
    #
    # So, what we actually end up doing, is:
    #
    #   %a: SymFloat(zf0) = placeholder[target=x]
    #
    # into
    #
    #   %na: f64[] = placeholder[target=x]
    #   %a: SymFloat(zf0) = call_function[target=torch.ops.higher_order.float_item](args = (zf0, %na,))
    #
    # After we have performed this transformation, we want to push float_item
    # as far down the graph as possible:
    # For example, convert:
    #
    #   %a: SymFloat(zf0) = call_function[target=torch.ops.higher_order.float_item](args = (zf0, %na,))
    #   %b: SymFloat(zf0 + 3) = call_function[target=operator.add](args = (%a, 3))
    #
    # into
    #
    #   %nb: f64[] = call_function[target=torch.add](args = (%na, 3))
    #   %b: SymFloat(zf0 + 3) = call_function[target=torch.ops.higher_order.float_item](args = (zf0 + 3, %na,))
    #
    # And we can eliminate it entirely if we hit a Tensor operation for which
    # we have a Tensor overload for the Scalar overload:
    #
    #   %a: SymFloat(zf0) = call_function[target=torch.ops.higher_order.float_item](args = (zf0, %na,))
    #   %b: f32[50] = call_function[target=torch.add](args = (%tensor, %a))
    #
    # into
    #
    #   %b: f32[50] = call_function[target=torch.add](args = (%tensor, %na))
    #
    # (I've omitted type promotion from the translations here for clarity.)
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

