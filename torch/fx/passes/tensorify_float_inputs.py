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

    # We proceed by a sequence of local transforms.  First, we mutate
    #
    # %a: SymFloat(s0) = placeholder[target=x]
    #
    # (TODO: should we give these s prefix in naming?)
    #
    # into
    #
    # %na: f64[] = placeholder[target=x]
    # %a: SymFloat(f0) = call_method[target=item](args = (%na,))
    #
    # Notice that after this transformation, we have introduced a new unbacked
    # float, and the original symbol s0 no longer has a 
    #
    #
    # If we subsequently guard on it, in a way that wasn't already
    # guarded upon in Dynamo prior to getting to this pass, this will now
    # cause code to fail to compile whereas previously it would have compiled.
    #
    #
    # Now, this is a bit unusual wrt unbacked handling.  Ordinarily,
    # subsequent guards on %a would fail guard on data dependent.  But these
    # are not truly data dependent: we know what their values are.
    #
    # This is handled via fake tensor constant propagation.  Because f64[] is
    # a constant 0d CPU tensor???


    # First, we find all SymFloat symbols which were used in a size-like way
    # (e.g., they show up in a size expression), and force specialization
    # on them.
    #
    # Hypothetically, we could support this in the future, but we
    # have to work harder: we're about to /delete/ the inputs that actually
    # bind these symbols, so you have to somehow replace them some other way.
    #
    # TODO: It would be marginally better to do this after DCE...

    # NB: This contains both float and int symbols, just cuz it's more
    # convenient to hoover them all up.  Distinguish them with var_to_val
    used_symbols = set()
    for node in graph.nodes:
        if node.op 
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

