from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from .symbolic_trace import symbolic_trace

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

        def attributes_are_equal(pn : Node, gn : Node) -> bool:
            # Use placeholder and output nodes as wildcards. The
            # only exception is that an output node can't match
            # a placeholder
            if (pn.op == "placeholder"
                    or (pn.op == "output" and gn.op != "placeholder")):
                return True
            return pn.op == gn.op and pn.target == gn.target

        # Terminate early if the node attributes are not equal
        if not attributes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        self.nodes_map[pn] = gn

        # Traverse the use-def relationships to ensure that `pn` is a true
        # match for `gn`
        if (pn.op != "output"
                and len(pn.all_input_nodes) != len(gn.all_input_nodes)):
            return False
        match_found = all(self._match_nodes(pn_, gn_) for pn_, gn_
                          in zip(pn.all_input_nodes, gn.all_input_nodes))
        if not match_found:
            self.nodes_map.pop(pn)
            return False

        return True


def replace_pattern(gm : GraphModule, pattern : Callable, replacement : Callable) -> None:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

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

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return torch.stack([w1, w2])

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing

    .. code-block:: python

        def pattern(x, y):
            return torch.neg(x) + torch.relu(y)

    with

    .. code-block:: python

        def replacement(x, y):
            return torch.relu(x)

    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.

    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:

    .. code-block:: python

        def forward(self, x, w1, w2):
            stack_1 = torch.stack([w1, w2])
            sum_1 = stack_1.sum()
            stack_2 = torch.stack([w1, w2])
            sum_2 = stack_2.sum()
            max_1 = torch.max(sum_1)
            add_1 = x + max_1
            max_2 = torch.max(sum_2)
            add_2 = add_1 + max_2
            return add_2

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
                for k, v in matcher.nodes_map.items():
                    # Shallow copy nodes_map
                    matches.append(Match(anchor=anchor,
                                   nodes_map=copy.copy(matcher.nodes_map)))

    # The set of all nodes in `original_graph` that we've seen thus far
    # as part of a pattern match
    replaced_nodes: Set[Node] = set()

    # Return TRUE if one of the nodes in the current match has already
    # been used as part of another match
    def overlaps_with_prev_match(match : Match) -> bool:
        for n in match.nodes_map.values():
            if n in replaced_nodes and n.op != "placeholder":
                return True
        return False

    for match in matches:

        # Skip overlapping matches
        if overlaps_with_prev_match(match):
            continue

        # Map replacement graph nodes to their copy in `original_graph`
        val_map: Dict[Node, Node] = {}

        pattern_placeholders = [n for n in pattern_graph.nodes
                                if n.op == "placeholder"]
        assert len(pattern_placeholders)
        replacement_placeholders = [n for n in replacement_graph.nodes
                                    if n.op == "placeholder"]
        assert len(pattern_placeholders) == len(replacement_placeholders)
        placeholder_map = {r : p for r, p
                           in zip(replacement_placeholders, pattern_placeholders)}

        # node from `original_graph` that matched with the output node
        # in `pattern`
        subgraph_output: Node = match.anchor

        def mark_node_as_replaced(n : Node) -> None:
            if n not in match.nodes_map.values():
                return
            for n_ in n.all_input_nodes:
                mark_node_as_replaced(n_)
            replaced_nodes.add(n)

        mark_node_as_replaced(subgraph_output)

        # Intialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        for replacement_node in replacement_placeholders:
            # Get the `original_graph` placeholder node
            # corresponding to the current `replacement_node`
            pattern_node = placeholder_map[replacement_node]
            original_graph_node = match.nodes_map[pattern_node]
            # Populate `val_map`
            val_map[replacement_node] = original_graph_node

        # Copy the replacement graph over
        with original_graph.inserting_before(subgraph_output):
            copied_output = original_graph.graph_copy(replacement_graph,
                                                      val_map)
        assert isinstance(copied_output, Node)

        # We only want to copy in the output node from `pattern` if we
        # have an output-output match. Otherwise, we leave out the
        # `pattern` output node so we don't have two outputs in the
        # resultant graph
        if subgraph_output.op != "output":
            subgraph_output = subgraph_output.args[0]    # type: ignore
        subgraph_output.replace_all_uses_with(copied_output)

        # Erase the `pattern` nodes
        for node in reversed(original_graph.nodes):
            if len(node.users) == 0 and node.op != "output":
                original_graph.erase_node(node)

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()
