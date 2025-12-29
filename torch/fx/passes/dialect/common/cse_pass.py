import itertools
from collections.abc import Callable, Hashable
from copy import deepcopy
from typing import Optional

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import Argument, Target
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten, TreeSpec


__all__ = ["get_CSE_banned_ops", "CSEPass"]

aten = torch.ops.aten


# stateful ops are banned from CSE
rand_ops = {
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
}  # noqa: E501,B950

inplace_ops = {
    aten.add_,
    aten.sub_,
    aten.mul_,
    aten.div_,
    aten.pow_,
    aten.lerp_,
    aten.relu_,
    aten.sigmoid_,
    aten.tanh_,
}  # noqa: E501


banned_ops = rand_ops.union(inplace_ops)


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def get_CSE_banned_ops() -> set:
    return banned_ops


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class CSEPass(PassBase):
    def __init__(self, is_impure_node: Optional[Callable[[Node], bool]] = None):
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, users only need to provide an ``is_impure_node``
        callable to indicate whether a node can be safely eliminated.

        Warning: CSE pass cannot be safely applied to FX graphs in non-functional
        dialects. If your dialect contains stateful operators, you must customize
        ``is_impure_node`` accordingly.
        """
        self.is_impure_node = is_impure_node
        super().__init__()

    def _find_impure_node(self, graph: Graph) -> set[Node]:
        """Find all impure nodes in a graph."""

        def get_aten_target(node: Node) -> Target:
            if hasattr(node.target, "overloadpacket"):
                return node.target.overloadpacket
            return node.target

        def is_impure(n: Node) -> bool:
            """Return True if a node should be considered impure."""
            if self.is_impure_node is not None:
                return self.is_impure_node(n)
            return n.is_impure() or get_aten_target(n) in banned_ops

        def check_args_kwargs_used_are_impure(n: Node) -> bool:
            """Check whether this node is impure due to its inputs being modified in-place.

            Example
            -------
            x = y + z
            y.add_(2)
            z = y + z

            In this case, ``x`` is considered impure since ``y`` is impure and
            cannot be merged with ``z``.
            """
            for arg in itertools.chain(n.args, n.kwargs.values()):
                if isinstance(arg, Node) and any(
                    is_impure(node_used) for node_used in list(arg.users)
                ):
                    return True
            return False

        impures = set()
        for n in graph.nodes:
            if is_impure(n):
                impures.add(n)
                if n.args and isinstance(n.args[0], Node):
                    # The first argument of an in-place operation becomes impure
                    # e.g., in y.add_(2), `y` is impure.
                    impures.add(n.args[0])
            if check_args_kwargs_used_are_impure(n):
                impures.add(n)
        return impures

    @staticmethod
    def _check_grad_node(n: Node, graph_grad_mode: bool) -> bool:
        """Return updated grad mode if the node changes it."""
        if n.target is torch._C._set_grad_enabled:
            assert isinstance(n.args[0], bool), (
                "_set_grad_enabled expects a bool argument."
            )
            graph_grad_mode = n.args[0]
        return graph_grad_mode

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph.

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """

        # Default to current graph grad mode
        graph_grad_mode = torch.is_grad_enabled()

        modified = False
        # Must use deepcopy to preserve hooks and pytree specs inside graph._codegen
        new_graph = deepcopy(graph_module.graph)
        csed_gm = GraphModule(graph_module, new_graph)

        impure_nodes = self._find_impure_node(new_graph)

        env: dict[
            Node, Node
        ] = {}  # map from node in the old graph to node in the new graph
        duplicated_node_to_replace: list[tuple[Node, int]] = []
        hash_node: dict[int, Node] = {}  # map from hash to a node in the new graph

        for n in new_graph.nodes:
            graph_grad_mode = CSEPass._check_grad_node(n, graph_grad_mode)

            # Do not eliminate placeholders, outputs, attrs, impure ops, or randomness
            if n.op in {"placeholder", "output", "get_attr"} or n in impure_nodes:
                continue
            else:
                # Flatten and substitute args and kwarg to be hashable
                def flatten_and_substitute(
                    arg_list: tuple[Argument, ...] | dict[str, Argument],
                ) -> tuple[Hashable, TreeSpec]:
                    arg_list, spec = tree_flatten(arg_list)
                    for i in range(len(arg_list)):
                        v = arg_list[i]
                        if isinstance(v, Node) and v in env:
                            # Substitute with an existing equivalent node
                            arg_list[i] = env[v]
                    return tuple(arg_list), spec

                args, args_spec = flatten_and_substitute(n.args)
                kwargs, kwargs_spec = flatten_and_substitute(n.kwargs)

                # Substitute nodes with the same hash
                hash_arg = (
                    n.op,
                    n.target,
                    args,
                    args_spec,
                    kwargs,
                    kwargs_spec,
                    graph_grad_mode,
                )
                hash_val = hash(hash_arg)

                # If a duplicate exists, substitute this node
                hash_val_in_hash_env = hash_val in hash_node
                if hash_val_in_hash_env:
                    # Delete and replace this duplicated node
                    duplicated_node_to_replace.append((n, hash_val))
                    env[n] = hash_node[hash_val]
                else:
                    hash_node[hash_val] = n

        if duplicated_node_to_replace:
            modified = True
            for duplicated_node, hash_val in duplicated_node_to_replace:
                duplicated_node.replace_all_uses_with(hash_node[hash_val])
                new_graph.erase_node(duplicated_node)

        csed_gm.recompile()
        return PassResult(csed_gm, modified)
