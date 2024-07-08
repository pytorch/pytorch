import logging
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any
import torch.fx as fx
from torch.fx._utils import lazy_format_graph_code
from torch.fx.graph_module import GraphModule

# TODO: refactor
from torch.fx.passes.runtime_assert import _get_sym_val

from torch.utils._sympy.reference import PythonReferenceAnalysis

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")

# The general shape of this transformation is to look for Tensor operations
# that take a backed SymFloat as an argument, and then redo them as tensor
# compute (with ints and tensors as inputs).  For example, add(Tensor, Scalar)
# can be translated into add(Tensor, Tensor).  Because Dynamo has already
# arranged for floats to be Tensor inputs to the graph, for typical float
# compute you can entirely translate the Python float operations into Tensor
# operations with only Tensor inputs.
#
# This pass is also responsible for doing CSE on the fly as we do this, since
# you don't want to keep recomputing the same quantity over and over again if
# it's used multiple times.
#
# This pass runs on the JOINT graph produced by AOT Autograd, prior to
# partitioning, because we want to be able make changes that affect our
# partitioning decisions (in particular, we want to avoid having to
# be able to save floats across the partition, and passes that change what
# device compute happen on need to happen before partitioning, but after this
# pass).  Note that some transformations have to happen before this in Dynamo,
# if fake tensor propagating the SymFloat would cause a spurious
# specialization.
#
# HISTORY NOTE:  Originally, I wanted to formulate this pass as pushing item()
# calls down, transforming float compute into int compute as we went.  If you
# manage to eliminate all float compute, this ends up being equivalent, but
# there is a critical difference when some floats cannot be eliminated: when
# we call item() on them, what should its SymFloat be?  Ideally, it would
# be the same backed SymFloat we had before.  But without symbolic expression
# propagation on tensor quantities, repropagating would instead give you an
# unbacked SymFloat.  Maybe it is a good idea to implement symbolic
# propagation on 0d scalar tensors, but I decided to go for something simpler
# to start.
#
# The boring stuff:
#
#  * What operators can I Tensor-ify?  (Anything with a Scalar argument)
#  * How do I Tensor-ify a SymFloat sympy expression (Sympy -> Op Handler ->
#    Tensor)


def tensorify_python_scalars(gm: GraphModule, shape_env: ShapeEnv):
    import sympy

    graph = gm.graph
    expr_to_sym_proxy = {}
    expr_to_tensor_proxy = {}

    first_non_placeholder = None
    placeholders = set()
    for node in graph.nodes:
        if node.op != "placeholder":
            first_non_placeholder = node
            break
        else:
            placeholders.add(node)

    Analysis = PythonReferenceAnalysis

    def _sympy_interp(expr):
        # sympy_interp() with hash consing
        from sympy import Integer, Number, Symbol
        from sympy.logic.boolalg import BooleanAtom

        from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp

        # hash cons
        if isinstance(expr, Symbol) and expr not in expr_to_tensor_proxy:
            # This is guaranteed to be populated by invariant established by
            # insert_deferred_runtime_asserts
            expr_to_tensor_proxy[expr] = fx.Proxy(
                graph.call_function(
                    torch.ops.aten.scalar_tensor.default,
                    (expr_to_sym_proxy[expr].node,),
                )
            )

        if expr in expr_to_tensor_proxy:
            return expr_to_tensor_proxy[expr]

        # base cases, don't cache
        if isinstance(expr, (Integer, Number, Symbol, BooleanAtom)):
            return sympy_interp(Analysis, expr_to_tensor_proxy, expr)

        # hash cons on arguments, run expr handler
        expr_to_tensor_proxy[expr] = _run_sympy_handler(
            Analysis,
            [_sympy_interp(arg) for arg in expr.args],
            expr,
        )
        return expr_to_tensor_proxy[expr]

    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else first_non_placeholder
        ):
            # Look for tensor.item() calls on placeholders
            if unbacked_bindings := node.meta.get("unbacked_bindings"):
                for s, keypath in unbacked_bindings.items():

                    def go(node, keypath):
                        if keypath == ():
                            return node
                        elif isinstance(keypath[0], CallMethodKey):
                            return go(
                                graph.call_method(keypath[0].name, (node,)), keypath[1:]
                            )
                        else:
                            return None

                    src_node = go(node, keypath)
                    if (
                        src_node.op == "call_function"
                        and src_node.target
                        is torch.ops.aten._local_scalar_dense.default
                    ):
                        # TODO: dtype conversion, so that we don't keep at too
                        # low precision
                        expr_to_tensor_proxy[s] = fx.Proxy(src_node.args[0])
                        expr_to_sym_proxy[s] = fx.Proxy(src_node)

            elif (sym_expr := _get_sym_val(node)) is not None:
                if sym_expr not in expr_to_sym_proxy and not isinstance(
                    sym_expr, (sympy.Number, sympy.logic.boolalg.BooleanAtom)
                ):
                    expr_to_sym_proxy[sym_expr] = fx.Proxy(node)

            # Look for functions to convert
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor:
                args = []
                transform = False
                for a in node.args:
                    if isinstance(a, fx.Node) and isinstance(
                        zf := a.meta["val"], torch.SymFloat
                    ):
                        transform = True
                        # TODO: populate meta on these
                        res = _sympy_interp(zf.node.expr).node
                        args.append(res)
                    else:
                        args.append(a)
                if transform:
                    res2 = graph.call_function(
                        torch.ops.aten.add.Tensor,
                        tuple(args),
                    )
                    node.replace_all_uses_with(res2, propagate_meta=True)
                    graph.erase_node(node)

    # DCE symbols (which are guaranteed to be pure) only
    for proxy in reversed(expr_to_sym_proxy.values()):
        if len(proxy.node.users) == 0:
            graph.erase_node(proxy.node)

    graph_code_log.debug(
        "%s",
        lazy_format_graph_code("tensorify_python_scalars", gm, colored=True),
    )
