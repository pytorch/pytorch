from __future__ import annotations

import logging
from typing import Union

import torch
import torch.fx as fx
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.symbolic_shapes import ShapeEnv  # noqa: TCH001
from torch.fx.graph_module import GraphModule  # noqa: TCH001

# TODO: refactor
from torch.fx.passes.runtime_assert import _get_sym_val
from torch.utils._sympy.reference import TensorReferenceAnalysis


__all__ = ["tensorify_python_scalars"]

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")

# The general shape of this transformation is to look for Tensor operations
# that take a backed SymFloat as an argument, and then redo them as tensor
# compute (with ints and tensors as inputs). For example, add(Tensor, Scalar)
# can be translated into add(Tensor, Tensor). Because Dynamo has already
# arranged for floats to be Tensor inputs to the graph, for typical float
# compute you can entirely translate the Python float operations into Tensor
# operations with only Tensor inputs.
#
# This pass is also responsible for doing CSE on the fly as we do this, since
# you don't want to keep recomputing the same quantity over and over again if
# it's used multiple times.
#
# This pass runs on the JOINT graph produced by AOT Autograd, prior to
# partitioning, because we want to be able to make changes that affect
# our partitioning decisions (in particular, we want to avoid having to
# be able to save floats across the partition, and passes that change what
# device compute happen on need to happen before partitioning, but after this
# pass). Note that some transformations have to happen before this in Dynamo,
# if fake tensor propagating the SymFloat would cause a spurious specialization.
#
# HISTORY NOTE: Originally, I wanted to formulate this pass as pushing item()
# calls down, transforming float compute into int compute as we went. If you
# manage to eliminate all float compute, this ends up being equivalent, but
# there is a critical difference when some floats cannot be eliminated: when
# we call item() on them, what should it's SymFloat be? Ideally, it would
# be the same backed SymFloat we had before. But without symbolic expresssion
# propogation on tensor quantities, repropagating would instead give you an
# unbacked SymFloat. Maybe it is a good idea to implement symbolic propagation
# on 0d scalar tensors, but I decided to go for something simpler to start.
#
# The boring stuff:
#
# * What operators can I Tensor-ify? (Anything with a Scalar argument)
# * How do I Tensor-ify a SymFloat sympy expression (Sympy -> Op Handler -> Tensor)
#
# TODO: make sure this runs before CPU->CUDA pass for cudagraph friendliness


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def tensorify_python_scalars(gm: GraphModule, shape_env: ShapeEnv) -> None:
    """
    Converts Python scalar operations into Tensor operations within the graph. This pass looks for
    Tensor operations that involve SymFloat arguments and transforms them into equivalent operations
    that use only Tensor inputs.

    Args:
        gm: The FX graph module representing the computation graph.
        shape_env: The shape environment responsible for symbolic shape tracking and propagation
        during graph transformations.

    Returns:
        None
    """
    import sympy

    graph = gm.graph
    tracer = fx.proxy.GraphAppendingTracer(graph)
    expr_to_sym_proxy: dict[sympy.Expr, fx.Proxy] = {}
    expr_to_tensor_proxy: dict[sympy.Expr, fx.Proxy] = {}

    first_non_placeholder = None
    placeholders = set()
    for node in graph.nodes:
        if node.op != "placeholder":
            first_none_placeholder = node
            break
        else:
            placeholders.add(node)

    Analysis = TensorReferenceAnalysis

    def _sympy_interp(expr: sympy.Expr) -> fx.Proxy:
        # sympy_interp() with hash consing, and special handling for
        # generating constants correctly
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
                ),
                tracer=tracer,
            )

        # cache constants, why not
        if isinstance(expr, (Integer, Number, BooleanAtom)):
            dtype = None
            c: Union[bool, int, float]
            if isinstance(expr, BooleanAtom):
                dtype = torch.bool
                c = bool(expr)
            elif isinstance(expr, sympy.Integer):
                dtype = torch.int64
                c = int(expr)
            elif isinstance(expr, sympy.Number):
                dtype = torch.float64
                c = float(expr)

            node = graph.call_function(
                torch.ops.aten.scalar_tensor.default, (c,), {"dtype": dtype}
            )
            node.meta["val"] = torch.tensor(c, dtype=dtype)
            expr_to_tensor_proxy[expr] = fx.Proxy(
                node,
                tracer=tracer,
            )

        if expr in expr_to_tensor_proxy:
            return expr_to_tensor_proxy[expr]

        # don't cache
        if isinstance(expr, Symbol):
            return sympy_interp(Analysis, expr_to_tensor_proxy, expr)  # type: ignore[arg-type]

        # hash cons on arguments, run expr handler
        expr_to_tensor_proxy[expr] = _run_sympy_handler(
            Analysis,
            [_sympy_interp(arg) for arg in expr.args],  # type: ignore[arg-type]
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
                for s in unbacked_bindings.keys():
                    if (
                        node is not None
                        and node.op == "call_function"
                        and node.target is torch.ops.aten._local_scalar_dense.default
                    ):
                        dtype = node.args[0].meta["val"].dtype
                        if dtype != torch.float64:
                            continue

                        assert isinstance(node.args[0], fx.Node), node.args[0]

                        expr_to_tensor_proxy[s] = fx.Proxy(node.args[0], tracer=tracer)
                        expr_to_sym_proxy[s] = fx.Proxy(node, tracer=tracer)

            elif (sym_expr := _get_sym_val(node)) is not None:
                if sym_expr not in expr_to_sym_proxy and not isinstance(
                    sym_expr, (sympy.Number, sympy.logic.boolalg.BooleanAtom)
                ):
                    expr_to_sym_proxy[sym_expr] = fx.Proxy(node, tracer=tracer)

            # Look for functions to convert
            if node.op == "call_function" and node.target is torch.ops.aten.mul.Tensor:
                args = []
                transform = False
                compute_dtype, result_dtype = elementwise_dtypes(
                    node.meta["val"],
                    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
                )

                for a in node.args:
                    if isinstance(a, fx.Node) and isinstance(
                        zf := a.meta["val"], torch.SymFloat
                    ):
                        transform = True
                        try:
                            a = _sympy_interp(zf.node.expr).node
                        except NotImplementedError:
                            transform = False
                            break

                        if "val" not in a.meta or a.meta["val"].dtype != compute_dtype:
                            res = graph.call_function(
                                torch.ops.prims.convert_element_type.default,
                                (
                                    a,
                                    compute_dtype,
                                ),
                            )
                            res.meta = node.meta
                            a = res

                        args.append(a)
                    else:
                        args.append(a)

                if transform:
                    res2 = graph.call_function(
                        torch.ops.aten.mul.Tensor,
                        tuple(args),
                    )

                    if compute_dtype != result_dtype:
                        res2 = graph.call_function(
                            torch.ops.prims.convert_element_type.default,
                            (
                                res2,
                                node.meta["val"].dtype,
                            ),
                        )

                    node.replace_all_uses_with(res2, propagate_meta=True)
                    graph.erase_node(node)

    # DCE symbols (which are guaranteed to be pure) only
    for proxy in reversed(expr_to_sym_proxy.values()):
        if len(proxy.node.users) == 0 and proxy.node.op != "placeholder":
            graph.erase_node(proxy.node)

    graph_code_log.debug(
        "%s", lazy_format_graph_code("tensorify_python_scalars", gm, colored=True)
    )
