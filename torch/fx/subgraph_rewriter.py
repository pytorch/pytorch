from torch.fx import Graph, GraphModule, Node, symbolic_trace

import copy
from typing import Callable, Dict, List, NamedTuple, Set

class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]

class SubgraphMatcher:
    def __init__(self, pattern : Graph) -> None:
        self.pattern = pattern
        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an "
                             "empty pattern")
        self.pattern_anchor = next(iter(reversed(pattern.nodes)))
        # Maps nodes in the pattern subgraph to nodes in the larger graph
        self.nodes_map: Dict[Node, Node] = {}

    def matches_subgraph_from_anchor(self, anchor : Node) -> bool:
        """
        Checks if the whole pattern can be matched starting from
        ``anchor`` in the larger graph.

        Pattern matching is done by recursively comparing the pattern
        node's use-def relationships against the graph node's.
        """
        self.nodes_map = {}
        return self._match_nodes(self.pattern_anchor, anchor)

    # Compare the pattern node `pn` against the graph node `gn`
    def _match_nodes(self, pn : Node, gn : Node) -> bool:
        # Check if we've already matched these nodes in the current
        # traversal
        if pn in self.nodes_map:
            return self.nodes_map[pn] == gn

        # We use "placeholder" nodes as wildcard matches but we don't
        # map them to any nodes in `original_graph`
        if pn.op == "placeholder":
            return True

        def attributes_are_equal(pn : Node, gn : Node) -> bool:
            if pn.op == "output":
                return True
            return pn.op == gn.op and pn.target == gn.target

        # Terminate early if the node attributes are not equal
        if not attributes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        self.nodes_map[pn] = gn

        # Traverse the use-def relationships to ensure that `pn` is a true
        # match for `gn`
        if ((not len(pn.all_input_nodes) and len(gn.all_input_nodes))
                or (len(pn.all_input_nodes) and not len(gn.all_input_nodes))):
            return False
        match_found = all(self._match_nodes(pn_, gn_) for pn_, gn_
                          in zip(pn.all_input_nodes, gn.all_input_nodes))
        if not match_found:
            self.nodes_map.pop(pn)
            return False

        return True


def replace_pattern(gm : GraphModule, pattern : Callable, replacement : Callable) -> None:
    """
    Matches a set of operators and their data dependencies (``pattern``)
    in the Graph of a GraphModule (``gm``), then replaces the matched
    subgraph with another set (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, w1, w2, b1, b2):
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.eq(torch.sum(t1), torch.sum(t2))

        def pattern(x, w1, w2, b1, b2):
            p1 = torch.cat([w1, w2])
            p2 = torch.cat([x, b2])
            return torch.addmm(b1, p1, p2.t())

        def replacement(x, w1, w2, b1, b2):
            m2 = torch.cat([x, b2])
            m1 = torch.cat([w1, w2])
            more_lines = torch.mm(w1, w2.t())
            return torch.addmm(b1, m1, m2.t())

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example,
    ``p1 = torch.cat([w1, w2])`` in ``pattern`` would match
    ``m1 = torch.cat([w1, w2])``. The ``return`` statement in
    ``pattern`` is matched based on its value only; it may or may not
    match to the ``return`` statement in the larger graph. In other
    words, the pattern doesn't have to extend to the end of the larger
    graph.

    Once the pattern is matched, it will be removed from the larger
    graph and replaced by ``replacement``. During this process, name
    mangling of ``replacement`` nodes may occur.

    Parameters can be added just like any other nodes. If you match
    ``forward(self, x, w1, w2)`` on ``pattern(self, x, w1, w2)`` with a
    replacement Callable ``replacement(x, w1, w2, b1, b2)``, then you'll
    end up with two additional parameters in your ``forward`` function:
    ``forward(self, x, w1, w2, b1, b2)``.
    """
    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph = gm.graph
    pattern_graph = symbolic_trace(pattern).graph
    replacement_graph = symbolic_trace(replacement).graph

    # Find all possible pattern matches in original_graph. Note that
    # pattern matches may overlap with each other.
    matcher = SubgraphMatcher(pattern_graph)
    matches: List[Match] = []

    # Consider each node as an "anchor" (deepest matching graph node)
    for anchor in original_graph.nodes:

        if matcher.matches_subgraph_from_anchor(anchor):

            def pattern_is_contained(nodes_map : Dict[Node, Node]) -> bool:
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
                                     nodes_map=copy.copy(matcher.nodes_map)))

    # The set of all nodes in `original_graph` that we've seen thus far
    # as part of a pattern match
    nodes_to_delete: Set[Node] = set()

    # Get the Nodes we'll use to hook the replacement subgraph in to the
    # original graph
    replacement_inputs: List[Node] = [n for n in replacement_graph.nodes
                                      if n.op == "placeholder"]

    # Return TRUE if one of the nodes in the current match has already
    # been used as part of another match
    def overlaps_with_prev_match(match : Match) -> bool:
        for n in match.nodes_map.values():
            if n in nodes_to_delete:
                return True
        return False

    for match in matches:

        # Skip overlapping matches
        if overlaps_with_prev_match(match):
            continue

        # Get the original graph nodes corresponding to the "start" and
        # "end" of the pattern subgraph. These connections are where
        # we'll hook the replacement subgraph in to the original graph
        original_graph_inputs: Dict[str, Node] = {n.target : n for n
                                                  in original_graph.nodes
                                                  if n.op == "placeholder"}
        subgraph_output: Node = match.anchor

        def mark_nodes_for_deletion(n : Node) -> None:
            if n not in match.nodes_map.values():
                return
            for n_ in n.all_input_nodes:
                mark_nodes_for_deletion(n_)
            nodes_to_delete.add(n)

        mark_nodes_for_deletion(subgraph_output)

        # Map replacement graph nodes to their copy in `original_graph`
        copied_nodes: Dict[Node, Node] = {}

        # Insert the new graph
        with original_graph.inserting_before(subgraph_output):
            last_placeholder = next(iter(reversed(original_graph_inputs.values())))
            for n in replacement_graph.nodes:
                if n.op == "output":
                    # To ensure that we have one and only one "output"
                    # node in our final graph, we need to change our
                    # logic slightly if the pattern extends to the end
                    # of the graph (i.e. if we matched the pattern
                    # "output" node to the original graph's "output"
                    # node)
                    if subgraph_output.op == "output":
                        # If we have an output-output match, we can just
                        # replace the "output" node in `original_graph`
                        # like we would any other node
                        copied_nodes[n] = original_graph.node_copy(n,
                                                                   lambda n :
                                                                   copied_nodes[n])
                        subgraph_output.replace_all_uses_with(copied_nodes[n])
                    else:
                        # If we don't have an output-output match, we
                        # need to keep `original_graph`'s existing
                        # "output" node. We update the existing output
                        # node's args and remove it from
                        # `nodes_to_delete`
                        assert len(n.all_input_nodes) == 1
                        n_input = next(iter(n.all_input_nodes))
                        new_args = [arg for arg in subgraph_output.args
                                    if arg not in nodes_to_delete]
                        delete_args = [arg for arg in subgraph_output.args
                                       if arg in nodes_to_delete]
                        new_args.append(copied_nodes[n_input])
                        new_kwargs = subgraph_output.kwargs
                        subgraph_output._update_args_kwargs(tuple(new_args), new_kwargs)
                        nodes_to_delete.remove(subgraph_output)
                        for arg in delete_args:
                            original_graph.erase_node(arg)
                            nodes_to_delete.remove(arg)
                elif n.op == "placeholder":
                    # If `n` corresponds to a placeholder node that
                    # already exists in `original_graph`, we don't want
                    # to swap it out
                    if n.target in original_graph_inputs.keys():
                        original_graph_node = original_graph_inputs[n.target]
                        if original_graph_node in nodes_to_delete:
                            nodes_to_delete.remove(original_graph_node)
                        copied_nodes[n] = original_graph_node
                    else:
                        # Temporarily change the insertion point to the
                        # beginning of the graph to ensure that placeholder
                        # nodes are inserted before anything else
                        with original_graph.inserting_after(last_placeholder):
                            copied_nodes[n] = original_graph.node_copy(n,
                                                                       lambda n :
                                                                       copied_nodes[n])
                            if n.name in original_graph_inputs.keys():
                                original_node = original_graph_inputs[n.name]
                                nodes_to_delete.add(original_node)
                                original_node.replace_all_uses_with(copied_nodes[n])
                            last_placeholder = copied_nodes[n]
                else:
                    copied_nodes[n] = original_graph.node_copy(n,
                                                               lambda n :
                                                               copied_nodes[n])

        def erase_nodes(n : Node) -> None:
            users = n.all_input_nodes
            if n in nodes_to_delete:
                original_graph.erase_node(n)
            for user in users:
                erase_nodes(user)

        original_graph_output = next(iter(reversed(original_graph.nodes)))
        erase_nodes(original_graph_output)

        # During graph replacement, we don't add `replacement`
        # placeholder nodes if a node by the same name already exists
        # in `original_graph`. This means that the we need to manually
        # update the placeholder nodes' "users" list to reflect the
        # current state of the graph. By updating placeholders from the
        # nodes that use them, we ensure that we have the correct
        # use-def relationships.
        for n in original_graph.nodes:
            for i in n.all_input_nodes:
                if i.op == "placeholder":
                    i.users[n] = None
                    users_to_remove = [user for user in i.users
                                       if user in list(nodes_to_delete)]
                    for user in users_to_remove:
                        i.users.pop(user)

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()
