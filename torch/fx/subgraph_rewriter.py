from torch.fx import Graph, Node, symbolic_trace

import collections
from typing import Callable, Dict, List, Set

# `anchor` - node from which the match was found
# `nodes_map` - maps nodes in the pattern subgraph to nodes in the
#               larger graph
Match = collections.namedtuple("Match", "anchor nodes_map")

class SubgraphMatcher:
    def __init__(self, pattern: Graph) -> None:
        self.pattern = pattern
        try:
            self.pattern_anchor = next(iter(reversed(pattern.nodes)))
        except Exception:
            raise ValueError("SubgraphMatcher cannot be initialized with an "
                             "empty pattern")
        # Maps nodes in the pattern subgraph to nodes in the larger graph
        self.nodes_map: Dict[Node, Node] = {}

    def matches_subgraph_from_anchor(self, anchor: Node) -> bool:
        """
        Check if the whole pattern can be matched starting from `anchor` in
        the larger graph
        """
        self.nodes_map = {}
        if (self.match_nodes(self.pattern_anchor, anchor)):
            return True
        return False

    def match_nodes(self, n1: Node, n2: Node) -> bool:
        """
        Compare Nodes `n1` (pattern node) and `n2` (graph node) for equality
        """
        # Check if we've already matched these nodes in the current
        # traversal
        if n1 in self.nodes_map:
            return self.nodes_map[n1] == n2

        # Terminate early if the node attributes are not equal
        if ((n1.op != n2.op and n1.op != "output")
            or (n1.op == "call_function" and n1.target != n2.target)
            or (n1.op == "placeholder" and n2.op == "placeholder"
                and n1.name != n2.name)):
            return False

        # Optimistically mark `n1` as a match for `n2`
        self.nodes_map[n1] = n2

        # Traverse the use-def relationships to ensure that `n1` is a true
        # match for `n2`
        for n1_ in n1.uses.keys():
            match_found = False
            for n2_ in n2.uses.keys():
                if match_found or self.match_nodes(n1_, n2_):
                    match_found = True
            if not match_found:
                self.nodes_map.pop(n1)
                return False

        return True


def replace_pattern(original_graph: Graph, pattern : Callable, replacement : Callable) -> None:
    """
    Function that takes one set of operators and their data dependencies
    and replaces them with another
    """

    from copy import copy

    # Get the graphs for `pattern` and `replacement`
    pattern_graph = symbolic_trace(pattern).graph
    replacement_graph = symbolic_trace(replacement).graph

    # Find all possible pattern matches in original_graph. Note that pattern
    # matches may overlap with each other.
    matcher = SubgraphMatcher(pattern_graph)
    matches: List[Match] = []

    # Consider each node as an "anchor" (deepest matching graph node)
    for anchor in original_graph.nodes:

        if matcher.matches_subgraph_from_anchor(anchor):

            def pattern_is_contained(nodes_map: Dict[Node, Node]) -> bool:
                # `lookup` represents all the nodes in `original_graph`
                # that are part of `pattern`
                lookup: Dict[Node, Node] = {v : k for k, v
                                            in nodes_map.items()}
                for n in lookup.keys():
                    if n.op == "placeholder" or lookup[n].op == "output":
                        continue
                    for user in n.users:
                        # If this node has users that were not in
                        # `lookup`, then it must leak out of the
                        # pattern subgraph
                        if user not in lookup:
                            return False
                return True

            # It's not a match if the pattern leaks out into the rest
            # of the graph
            if pattern_is_contained(matcher.nodes_map):
                # Shallow copy nodes_map
                matches.append(Match(anchor=anchor,
                                     nodes_map=copy(matcher.nodes_map)))

    # The set of all nodes in `original_graph` that we've seen thus far
    # as part of a pattern match
    nodes_to_delete: Set[Node] = set()

    # Get the Nodes we'll use to hook the replacement subgraph in to the
    # original graph
    replacement_inputs: List[Node] = [n for n in replacement_graph.nodes
                                      if n.op == "placeholder"]
    replacement_output: Node = next(n for n in replacement_graph.nodes
                                    if n.op == "output")

    # Return TRUE if one of the nodes in the current match has already
    # been used as part of another match
    def overlaps_with_prev_match(match: Match) -> bool:
        for n in match.nodes_map.values():
            if n in nodes_to_delete:
                return True
        return False

    if not matches:
        raise ValueError("No matches were found for the given pattern")

    for match in matches:

        # Skip overlapping matches
        if overlaps_with_prev_match(match):
            continue

        # Get the original graph nodes corresponding to the "start" and
        # "end" of the pattern subgraph. These connections are where
        # we'll hook the replacement subgraph in to the original graph
        original_graph_inputs: Dict[str, Node] = {n.name : n
                                                  for n
                                                  in original_graph.nodes
                                                  if n.op == "placeholder"}
        subgraph_output: Node = match.anchor

        def mark_nodes_for_deletion(n: Node) -> None:
            if n not in match.nodes_map.values():
                return
            for n_ in n.uses:
                mark_nodes_for_deletion(n_)
            nodes_to_delete.add(n)
            # By removing the `original_graph` node's name early, we
            # ensure that we can insert nodes from `replacement` with
            # the same name. (This is strictly cosmetic)
            if n.name in original_graph._used_names:
                original_graph._used_names.pop(n.name)

        mark_nodes_for_deletion(subgraph_output)

        # Map replacement graph nodes to their copy in `original_graph`
        copied_nodes: Dict[Node, Node] = {}

        # Insert the new graph
        with original_graph.inserting_before(subgraph_output):
            for n in replacement_graph.nodes:
                if n.op == "output":
                    # To ensure that we have one and only one "output"
                    # node in our final graph, we need to change our
                    # logic slightly if the pattern extends to the end
                    # of the graph (i.e. if we matched the pattern
                    # "output" node to the original graph's "output"
                    # node)
                    if subgraph_output.op != "output":
                        n = n.args[0]
                        nodes_to_delete.add(copied_nodes[n])
                    copied_nodes[n] = original_graph.node_copy(n,
                                                               lambda n :
                                                               copied_nodes[n])
                    subgraph_output.replace_all_uses_with(copied_nodes[n])
                elif n.op == "placeholder":
                    # Temporarily change the insertion point to the
                    # beginning of the graph to ensure that placeholder
                    # nodes are inserted before anything else
                    with original_graph.inserting_before(None):
                        copied_nodes[n] = original_graph.node_copy(n,
                                                                   lambda n :
                                                                   copied_nodes[n])
                        if n.name in original_graph_inputs.keys():
                            original_node = original_graph_inputs[n.name]
                            nodes_to_delete.add(original_node)
                            original_node.replace_all_uses_with(copied_nodes[n])
                            original_node = copied_nodes[n]
                else:
                    copied_nodes[n] = original_graph.node_copy(n,
                                                               lambda n :
                                                               copied_nodes[n])

        for n in list(nodes_to_delete):
            n._remove_from_list()
            n._erased = True
            original_graph._len -= 1
