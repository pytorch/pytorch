import torch
from typing import Any, Dict, Optional, Set, TYPE_CHECKING
import logging
if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any
from torch.fx.graph_module import GraphModule
from torch.fx.experimental.sym_node import SymNode
from torch.fx._utils import lazy_format_graph_code
import torch.fx as fx

# TODO: refactor
from torch.fx.passes.runtime_assert import _get_example_value

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
    from torch.utils._sympy.interp import sympy_interp
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    symbol_to_proxy = {}
    placeholders = set()
    last_placeholder = None
    graph = gm.graph
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        last_placeholder = node
        placeholders.add(node)
    if last_placeholder is None:  # no placeholders, just insert before first node
        last_placeholder = next(iter(graph.nodes))
    symbol_to_tensor_proxy = {}

    # Detect bindings: TODO refactor
    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else last_placeholder.next
        ):
            # Find s0 symbols
            if (
                node in placeholders
                and (example_value := _get_example_value(node)) is not None
            ):
                def match_symbol(symint, cb):
                    if (
                        isinstance(symint, torch.SymInt)
                        and isinstance(symint.node, SymNode)
                        and isinstance(s := symint.node.expr, sympy.Symbol)
                        and s not in symbol_to_proxy
                    ):
                        symbol_to_proxy[s] = fx.Proxy(cb())

                match_symbol(example_value, lambda: node)

                # TODO: consider looking for the other bindings too

            # Find u0/zf0 symbols
            if unbacked_bindings := node.meta.get("unbacked_bindings"):
                for s, keypath in unbacked_bindings.items():
                    # TODO: some CSE when generating these nodes can probably
                    # help reduce graph size and improve compile time
                    def go(node, keypath):
                        if keypath == ():
                            return node
                        if (
                            len(keypath) >= 2
                            and isinstance(keypath[0], CallMethodKey)
                            and isinstance(keypath[1], pytree.SequenceKey)
                        ):
                            if keypath[0].name == "size":
                                return go(
                                    graph.call_function(
                                        torch.ops.aten.sym_size.int,
                                        (node, keypath[1].idx),
                                    ),
                                    keypath[2:],
                                )
                            if keypath[0].name == "stride":
                                return go(
                                    graph.call_function(
                                        torch.ops.aten.stride.int,
                                        (node, keypath[1].idx),
                                    ),
                                    keypath[2:],
                                )
                            return go(
                                graph.call_method(
                                    keypath[0].name, (node, keypath[1].idx)
                                ),
                                keypath[2:],
                            )
                        elif isinstance(keypath[0], CallMethodKey):
                            return go(
                                graph.call_method(keypath[0].name, (node,)), keypath[1:]
                            )
                        elif isinstance(keypath[0], pytree.SequenceKey):
                            return go(
                                graph.call_function(
                                    operator.getitem, (node, keypath[0].idx)
                                ),
                                keypath[1:],
                            )
                        elif isinstance(keypath[0], ConvertIntKey):
                            return go(
                                graph.call_function(
                                    cast_symbool_to_symint_guardless, (node,)
                                ),
                                keypath[1:],
                            )
                        elif isinstance(keypath[0], DivideByKey):
                            # TODO: need to assert divisibility
                            return go(
                                graph.call_function(
                                    operator.floordiv, (node, keypath[0].divisor)
                                ),
                                keypath[1:],
                            )
                        elif isinstance(keypath[0], InnerTensorKey):
                            return go(
                                graph.call_function(
                                    getattr, (node, keypath[0].inner_name)
                                ),
                                keypath[1:],
                            )
                        else:
                            raise AssertionError(f"unrecognized keypath {keypath}")

                    src_node = go(node, keypath)
                    if src_node.op == "call_function" and src_node.target is torch.ops.aten._local_scalar_dense.default:
                        # TODO: dtype conversion, so that we don't keep at too
                        # low precision
                        symbol_to_tensor_proxy[s] = fx.Proxy(src_node.args[0])
                    else:
                        symbol_to_proxy[s] = fx.Proxy(src_node)

            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.add.Tensor
            ):
                args = []
                transform = False
                for a in node.args:
                    if isinstance(a, fx.Node) and isinstance(zf := a.meta["val"], torch.SymFloat):
                        transform = True
                        # TODO: populate meta on these
                        res = sympy_interp(
                            # TODO: this reference analysis is wrong, want the
                            # Tensor reference analysis
                            PythonReferenceAnalysis,
                            symbol_to_tensor_proxy,
                            zf.node.expr
                        ).node
                        args.append(res)
                    else:
                        args.append(a)
                if transform:
                    res2 = graph.call_function(
                        torch.ops.aten.add.Tensor,
                        tuple(args),
                    )
                    node.replace_all_uses_with(res2, propagate_meta=True)
    print("all done")

    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(f"tensorify_python_scalars", gm),
    )
