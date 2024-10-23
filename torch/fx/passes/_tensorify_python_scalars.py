from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

from sympy import Integer, Number, Symbol
from sympy.logic.boolalg import BooleanAtom

import torch
import torch.fx as fx
from torch._prims_common import get_computation_dtype
from torch._subclasses import fake_tensor  # noqa: TCH001
from torch._utils_internal import JustKnobsConfig
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.symbolic_shapes import ShapeEnv  # noqa: TCH001
from torch.fx.graph_module import GraphModule  # noqa: TCH001

# TODO: refactor
from torch.fx.passes.runtime_assert import _get_sym_val
from torch.fx.proxy import MetaProxy
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import TensorReferenceAnalysis


__all__: List[str] = []

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
# This pass runs on the JOINT graph produced by AOT Autograd, prior to partitioning.
# The primary goal of this pass is to eliminate floats by replacing TensorScalar
# operations with TensorTensor operations and then Dead Code Elimination (DCE) of
# the item calls, which effectively removes the floats.
#
# This needs to happen before partitioning because it influences partitioning decisions,
# specifically by ensuring that we don't need to save floats across partitions.
# Additionally, there is a separate pass that changes which device computations
# occur on. That pass must be run after this one, but still before partitioning.
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


SUPPORTED_OPS = {
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.div.Tensor,
}


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def tensorify_python_scalars(
    gm: GraphModule, shape_env: ShapeEnv, fake_mode: fake_tensor.FakeTensorMode
) -> None:
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

    knob = JustKnobsConfig(
        name="pytorch/compiler:tensorify_python_scalars",
        env_name="TENSORIFY_PYTHON_SCALARS",
        default=True,
    ).get()
    if not knob:
        return None

    graph = gm.graph
    tracer = fx.proxy.GraphAppendingTracer(graph)
    expr_to_sym_proxy: dict[sympy.Expr, MetaProxy] = {}
    expr_to_tensor_proxy: dict[sympy.Expr, MetaProxy] = {}
    specialized_float_nodes: dict[fx.Node, float] = {}
    deleted: set[fx.Node] = set()

    first_non_placeholder = None
    placeholders = set()
    for node in graph.nodes:
        if node.op != "placeholder":
            first_none_placeholder = node
            break
        else:
            placeholders.add(node)

    Analysis = TensorReferenceAnalysis

    def dce(graph: fx.Graph, deleted: set[fx.Node]) -> None:
        for node in graph.nodes:
            if (
                len(node.users) == 0
                and node.op != "placeholder"
                and node.op != "output"
                and node not in deleted
            ):
                graph.erase_node(node)
                deleted.add(node)

    def _sympy_interp(expr: sympy.Expr) -> MetaProxy:
        # sympy_interp() with hash consing, and special handling for
        # generating constants correctly

        # hash cons
        if isinstance(expr, Symbol) and expr not in expr_to_tensor_proxy:
            # This is guaranteed to be populated by invariant established by
            # insert_deferred_runtime_asserts
            expr_to_tensor_proxy[expr] = torch.ops.aten.scalar_tensor.default(
                expr_to_sym_proxy[expr]
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
            with fake_mode:
                node.meta["val"] = torch.ops.aten.scalar_tensor.default(c, dtype=dtype)
            expr_to_tensor_proxy[expr] = MetaProxy(
                node,
                tracer=tracer,
                fake_mode=fake_mode,
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
            if (
                node is not None
                and node.op == "call_function"
                and node.target is torch.ops.aten._local_scalar_dense.default
            ):
                dtype = node.args[0].meta["val"].dtype
                if dtype != torch.float64:
                    continue

                assert isinstance(node.args[0], fx.Node), node.args[0]

                s = node.meta["val"].node.expr
                expr_to_tensor_proxy[s] = MetaProxy(
                    node.args[0], tracer=tracer, fake_mode=fake_mode
                )
                expr_to_sym_proxy[s] = MetaProxy(
                    node, tracer=tracer, fake_mode=fake_mode
                )

            elif (sym_expr := _get_sym_val(node)) is not None:
                if sym_expr not in expr_to_sym_proxy and not isinstance(
                    sym_expr, (sympy.Number, sympy.logic.boolalg.BooleanAtom)
                ):
                    expr_to_sym_proxy[sym_expr] = MetaProxy(
                        node, tracer=tracer, fake_mode=fake_mode
                    )

            # Look for functions to convert
            if node.op == "call_function" and node.target in SUPPORTED_OPS:
                args: List[Any] = []
                transform = False
                compute_dtype = get_computation_dtype(node.meta["val"].dtype)

                for a in node.args:
                    if (
                        isinstance(a, fx.Node)
                        and "val" in a.meta
                        and isinstance(zf := a.meta["val"], torch.SymFloat)
                    ):
                        transform = True
                        try:
                            proxy = _sympy_interp(zf.node.expr)
                        except NotImplementedError:
                            transform = False
                            break

                        if proxy.node.meta["val"].dtype != compute_dtype:
                            proxy = torch.ops.prims.convert_element_type.default(
                                proxy, compute_dtype
                            )

                        args.append(proxy)
                    elif isinstance(a, fx.Node):
                        args.append(MetaProxy(a, tracer=tracer, fake_mode=fake_mode))
                    else:
                        args.append(a)

                if transform:
                    replacement_proxy = node.target(*args)

                    if compute_dtype != node.meta["val"].dtype:
                        replacement_proxy = (
                            torch.ops.prims.convert_element_type.default(
                                replacement_proxy,
                                node.meta["val"].dtype,
                            )
                        )

                    node.replace_all_uses_with(replacement_proxy.node)
                    graph.erase_node(node)
                    deleted.add(node)

    # DCE symbols (which are guaranteed to be pure) only
    dce(graph, deleted)

    # Now do one more pass that specializes all symfloats we didn't manage
    # to tensorify away.
    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else first_non_placeholder
        ):
            args = []
            kwargs: Dict[str, Any] = {}
            transform = False
            for a in node.args:
                if a in specialized_float_nodes:
                    args.append(specialized_float_nodes[a])
                elif (
                    isinstance(a, fx.Node)
                    and "val" in a.meta
                    and isinstance(zf := a.meta["val"], torch.SymFloat)
                    and isinstance(zf.node.expr, Symbol)
                ):
                    transform = True
                    args.append(float(zf))
                else:
                    args.append(a)

            for k, v in node.kwargs.items():
                if (
                    isinstance(v, fx.Node)
                    and "val" in v.meta
                    and isinstance(zf := v.meta["val"], torch.SymFloat)
                    and isinstance(zf.node.expr, Symbol)
                ):
                    transform = True
                    kwargs[k] = float(zf)
                else:
                    kwargs[k] = v

            if (
                node.op != "placeholder"
                and node.op != "get_attr"
                and all(isinstance(arg, (float)) for arg in args)
                and (type(result := node.target(*args, **kwargs)) == float)
            ):
                # If all args are constants and resulting value is a float,
                # add node to specialized_float_nodes and continue.
                # In later iterations we'll DCE this node by specializing the float value
                # into the args of downstream fx nodes.
                specialized_float_nodes[node] = result
                continue

            if transform:
                # args may not be metaproxy (eg. float) so can't use metaproxy here
                replacement_node = graph.call_function(node.target, tuple(args), kwargs)
                replacement_node.meta["val"] = node.target(*args, **kwargs)
                node.replace_all_uses_with(replacement_node)
                graph.erase_node(node)

    # DCE one more time to remove specialized item calls
    dce(graph, deleted)

    graph_code_log.debug(
        "%s", lazy_format_graph_code("tensorify_python_scalars", gm, colored=True)
    )
