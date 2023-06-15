import logging
import operator
from typing import List

import networkx

import torch
from torch._dynamo.utils import counters

from ..pattern_matcher import (
    CallFunctionVarArgs,
    config_flag,
    get_arg_value,
    get_mutation_region_id,
    stable_topological_sort,
)

log = logging.getLogger(__name__)


def get_nx_graph_from_fx_graph(graph: torch.fx.Graph) -> networkx.DiGraph:
    nx_graph = networkx.DiGraph()
    for node in graph.nodes:
        nx_graph.add_node(node)
        for user in node.users:
            nx_graph.add_edge(node, user)
    return nx_graph


class GroupFusionPass:
    """
    `GroupFusionPass` implements an Fx pass which groups together similar nodes into a fused op. Unlike `PatternMatcherPass`
    which can only match local patterns in a graph, this pass can group together nodes across the graph. The only restriction
    imposed is that the nodes being fused must not have a path in between each other (which would lead to a cycle). Additional
    checks such as `extra_check` (at a node level), or `pair_check` can be added to pose additional restrictions on
    nodes to be fused.
    """

    def __init__(
        self,
        pattern,
        pair_check,
        replacement_fn,
        extra_check=lambda m: True,
        *,
        prevent_match_across_mutations,
    ):
        """
        Args:
            pattern (torch.fx.Graph): A pattern to match nodes against. this pattern is used to filter nodes to be fused
            pair_check (Callable[[torch.fx.Node, torch.fx.Node], bool]): A function to check if a pair of nodes can be fused
            replacement_fn (Callable[[torch.fx.Graph, List[torch.fx.Node]]]): Function which does the actual replacement
            extra_check (Callable[[torch.fx.Node], bool]): Extra check to filter the nodes
            prevent_match_across_mutations (bool): Prevent matching across mutations.
        """
        self.pattern = pattern
        self.extra_check = extra_check
        self.pair_check = pair_check
        self.replacement_fn = replacement_fn
        self.prevent_match_across_mutations = prevent_match_across_mutations

    def apply(self, graph):
        target_nodes = []
        for node in graph.nodes:
            if m := self.pattern.match(node):
                if self.extra_check(m):
                    target_nodes.append(node)

        if len(target_nodes) <= 1:
            return

        seen_nodes = set()
        nx_graph = get_nx_graph_from_fx_graph(graph)

        for i, target_node in enumerate(target_nodes):
            nodes_to_fuse = []
            if target_node not in seen_nodes:
                seen_nodes.add(target_node)
                nodes_to_fuse.append(target_node)

                for j in range(i + 1, len(target_nodes)):
                    if target_nodes[j] in seen_nodes:
                        continue
                    can_be_fused = self.pair_check(target_node, target_nodes[j]) and (
                        not self.prevent_match_across_mutations
                        or get_mutation_region_id(graph, target_node)
                        == get_mutation_region_id(graph, target_nodes[j])
                    )
                    if (
                        can_be_fused
                    ):  # Check no conflict with any other nodes being fused
                        for node_to_fuse in nodes_to_fuse:
                            if networkx.has_path(
                                nx_graph, node_to_fuse, target_nodes[j]
                            ) or networkx.has_path(
                                nx_graph, target_nodes[j], node_to_fuse
                            ):
                                can_be_fused = False
                                break
                    if can_be_fused:
                        nodes_to_fuse.append(target_nodes[j])
                        seen_nodes.add(target_nodes[j])
                if len(nodes_to_fuse) == 1:
                    continue
                self.replacement_fn(graph, nodes_to_fuse)
                nx_graph = get_nx_graph_from_fx_graph(graph)


def layer_norm_replacement(graph: torch.fx.Graph, nodes_to_fuse: List[torch.fx.Node]):
    """
    Replace multiple `layer_norm` with a single `layer_norm`.
    """
    inputs = []
    shapes = []
    weights = []
    biases = []
    epss = []

    for ln in nodes_to_fuse:
        inputs.append(get_arg_value(ln, 0, "input"))
        shapes.append(get_arg_value(ln, 1, "normalized_shape"))
        weights.append(get_arg_value(ln, 2, "weight"))
        biases.append(get_arg_value(ln, 3, "bias"))
        eps = get_arg_value(ln, 4, "eps")
        if eps is None:
            eps = 1e-5
        epss.append(eps)
        counters["inductor"]["layer_norm_removed"] += 1

    stack_dim = -1 - len(shapes[-1])

    with graph.inserting_before(nodes_to_fuse[0]):
        # Stack inputs
        stack_input = graph.call_function(torch.stack, args=(inputs, stack_dim))

        # Stack weight
        stack_weight = graph.call_function(torch.stack, args=(weights,))

        # Stack bias
        stack_bias = graph.call_function(torch.stack, args=(biases,))

        group_layer_norm = graph.call_function(
            torch.nn.functional.layer_norm,
            args=(stack_input, shapes[-1]),
            kwargs={"eps": epss[-1]},
        )

        group_layer_norm = graph.call_function(
            torch.addcmul, args=(stack_bias, stack_weight, group_layer_norm)
        )

        group_layer_norm = graph.call_function(
            torch.unbind, args=(group_layer_norm,), kwargs={"dim": stack_dim}
        )

        counters["inductor"]["layer_norm_added"] += 1
        for i, ln in enumerate(nodes_to_fuse):
            getitem = graph.call_function(operator.getitem, args=(group_layer_norm, i))
            ln.replace_all_uses_with(getitem)
            getitem.meta.update(ln.meta)

        for ln in nodes_to_fuse:
            graph.erase_node(ln)

        log.info("Fused %d layer norms", len(nodes_to_fuse))

        stable_topological_sort(graph)


def layer_norm_pair_check(ln1, ln2):
    """
    Check if two `layer_norm` nodes can be fused
    """
    return (
        ln1.meta["example_value"].shape == ln2.meta["example_value"].shape
        and get_arg_value(ln1, 1, "normalized_shape")
        == get_arg_value(ln2, 1, "normalized_shape")
        and get_arg_value(ln1, 4, "eps") == get_arg_value(ln2, 4, "eps")
    )


def layer_norm_extra_check(ln):
    # Current implementation requires weight and bias to be present
    return bool(get_arg_value(ln, 2, "weight")) and bool(get_arg_value(ln, 3, "bias"))


layer_norm_fusion_pass = GroupFusionPass(
    pattern=CallFunctionVarArgs(torch.nn.functional.layer_norm),
    extra_check=lambda m: config_flag("group_fusion_fx_passes")(m)
    and layer_norm_extra_check(m),
    pair_check=layer_norm_pair_check,
    replacement_fn=layer_norm_replacement,
    prevent_match_across_mutations=True,
)
