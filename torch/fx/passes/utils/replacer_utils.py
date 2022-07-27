import copy
import torch
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx import symbolic_trace, GraphModule, Node
from typing import Dict, List, Callable, Optional
from torch.fx.passes.tools_common import legalize_graph


def _replace_submodules(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_submodule(mod: torch.nn.Module, target: str) -> Optional[torch.nn.Module]:
        try:
            mod_match = mod.get_submodule(target)
            return mod_match
        except AttributeError:
            return None

    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":

            gm_submod = try_get_submodule(gm, node.target)

            replacement_submod = try_get_submodule(replacement, node.target)

            # CASE 1: This target already exists as a submodule in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_submod is not None:
                continue

            # CASE 2: The target exists as a submodule in `replacement`
            # only, so we need to copy it over.
            elif replacement_submod is not None:
                new_submod = copy.deepcopy(getattr(replacement, node.target))
                gm.add_submodule(node.target, new_submod)

            # CASE 3: The target doesn't exist as a submodule in `gm`
            # or `replacement`
            else:
                raise RuntimeError("Attempted to create a \"", node.op,
                                   "\" node during subgraph rewriting "
                                   f"with target {node.target}, but "
                                   "the referenced submodule does not "
                                   "exist in either the original "
                                   "GraphModule `gm` or the replacement"
                                   " GraphModule `replacement`")

    gm.graph.lint()

def replace_pattern(gm: GraphModule, pattern: Callable, replacement: Callable):

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph = gm.graph
    pattern_graph = symbolic_trace(pattern).graph
    replacement_graph = symbolic_trace(replacement).graph

    matcher = SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False, fully_contained=True)
    matches = matcher.match(original_graph)

    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == "placeholder"]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = dict()

    for match in matches:
        # TODO: avoid overlapping matchs

        # TODO: this is wrong to depends on
        # node from `original_graph` that matched with the output node  in `pattern`
        # subgraph_output: Node = match.returning_nodes[0].

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            val_map[rn] = match_changed_node.get(gn, gn)

        # Copy the replacement graph over
        # with original_graph.inserting_before(subgraph_output):
        copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes, )

        gm.recompile()
        print("after")
        original_graph.print_tabular()

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node

        gm.recompile()
        print("\nafter output replacement")
        gm.graph.print_tabular()

        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        gm.recompile()
        print("\nafter cleanup")
        gm.graph.print_tabular()

    # TODO: avoid calling this
    legalize_graph(gm)
    print("\nafter legalize_graph")
    gm.graph.print_tabular()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_submodules(gm, replacement)

    return matches
