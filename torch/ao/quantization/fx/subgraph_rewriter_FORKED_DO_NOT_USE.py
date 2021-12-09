from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx._compatibility import compatibility

import copy
from typing import Callable, Dict, List, NamedTuple, Optional, Set
import torch

@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]

class _SubgraphMatcher:
    def __init__(self, pattern: Graph) -> None:
        self.pattern = pattern
        if len(pattern.nodes) == 0:
            raise ValueError("_SubgraphMatcher cannot be initialized with an "
                             "empty pattern")
        # `self.pattern_anchor` is the output Node in `pattern`
        self.pattern_anchor = next(iter(reversed(pattern.nodes)))
        # Ensure that there is only a single output value in the pattern
        # since we don't support multiple outputs
        assert len(self.pattern_anchor.all_input_nodes) == 1, \
            "Pattern matching on multiple outputs is not supported"
        # Maps nodes in the pattern subgraph to nodes in the larger graph
        self.nodes_map: Dict[Node, Node] = {}

    def matches_subgraph_from_anchor(
            self,
            anchor: Node,
            original_module: GraphModule,
            pattern_module: GraphModule) -> bool:
        """
        Checks if the whole pattern can be matched starting from
        ``anchor`` in the larger graph.

        Pattern matching is done by recursively comparing the pattern
        node's use-def relationships against the graph node's.
        """
        self.nodes_map = {}
        return self._match_nodes(
            self.pattern_anchor, anchor, original_module, pattern_module)

    # Compare the pattern node `pn` against the graph node `gn`
    def _match_nodes(
            self,
            pn: Node,
            gn: Node,
            original_module: GraphModule,
            pattern_module: GraphModule) -> bool:

        if isinstance(pn, (tuple, list)):
            if not isinstance(gn, type(pn)):
                return False
            return all(self._match_nodes(a1, a2, original_module, pattern_module) for
                       a1, a2 in zip(pn, gn))  # type: ignore[call-overload]
        elif isinstance(pn, dict):
            if not isinstance(gn, dict):
                return False
            return pn.keys() == gn.keys() and \
                all(self._match_nodes(v1, v2, original_module, pattern_module)
                    for v1, v2 in zip(pn.values(), gn.values()))

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in self.nodes_map:
            return self.nodes_map[pn] == gn

        PRIM_TYPES = (int, float, torch.dtype)
        # if both pattern and graph are not Node, we check for equality of these values
        if not isinstance(pn, Node) and not isinstance(gn, Node):
            return pn == gn

        # trying to match the input in pattern graph with a primitive type values
        if isinstance(gn, PRIM_TYPES):
            if isinstance(pn, Node) and pn.op == "placeholder":
                self.nodes_map[pn] = gn
                return True
            else:
                return False


        original_modules = dict(original_module.named_modules())
        pattern_modules = dict(pattern_module.named_modules())

        def attributes_are_equal(pn: Node, gn: Node) -> bool:
            # Use placeholder and output nodes as wildcards. The
            # only exception is that an output node can't match
            # a placeholder
            if (pn.op == "placeholder"
                    or (pn.op == "output" and gn.op != "placeholder")):
                return True
            elif pn.op == "get_attr" and gn.op == "get_attr":
                # assuming get_attr nodes are the same
                return True
            elif pn.op == "call_module" and gn.op == "call_module":
                original_m = original_modules[gn.target]
                pattern_m = pattern_modules[pn.target]
                return type(original_m) == type(pattern_m)
            return pn.op == gn.op and pn.target == gn.target

        # Terminate early if the node attributes are not equal
        if not attributes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        self.nodes_map[pn] = gn

        # Traverse the use-def relationships to ensure that `pn` is a true
        # match for `gn`
        if pn.op == "placeholder":
            return True
        if (pn.op != "output"
                and len(pn.args) != len(gn.args)):
            return False
        if pn.op == "output":
            match_found = any(self._match_nodes(pn.all_input_nodes[0], gn_, original_module, pattern_module)
                              for gn_ in gn.all_input_nodes)
        else:
            # using args here to make sure we can match Node and non-Node
            # arguments
            # also allows us to match a Node with a primitive type value
            match_found = (len(pn.args) == len(gn.args)
                           and all(self._match_nodes(pn_, gn_, original_module, pattern_module)  # type: ignore[arg-type]
                                   for pn_, gn_ \
                                   in zip(pn.args, gn.args)))
        if not match_found:
            self.nodes_map.pop(pn)
            return False

        return True


def _replace_submodules(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_submodule_or_attr(mod: torch.nn.Module, target: str) -> Optional[torch.nn.Module]:
        try:
            mod_match = mod.get_submodule(target)
            return mod_match
        except AttributeError:
            pass

        # supports getattr as well
        try:
            attr = getattr(mod, target)
            return attr
        except AttributeError:
            return None


    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":

            gm_submod = try_get_submodule_or_attr(gm, node.target)

            replacement_submod = try_get_submodule_or_attr(replacement, node.target)

            # CASE 1: This target already exists as a submodule in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_submod is not None:
                continue

            # CASE 2: The target exists as a submodule in `replacement`
            # only, so we need to copy it over.
            elif replacement_submod is not None:
                new_submod = copy.deepcopy(getattr(replacement, node.target))
                if isinstance(new_submod, torch.nn.Module):
                    gm.add_submodule(node.target, new_submod)
                else:
                    setattr(gm, node.target, new_submod)

            # CASE 3: The target doesn't exist as a submodule in `gm`
            # or `replacement`
            else:
                continue
                raise RuntimeError("Attempted to create a \"", node.op,
                                   "\" node during subgraph rewriting "
                                   f"with target {node.target}, but "
                                   "the referenced submodule does not "
                                   "exist in either the original "
                                   "GraphModule `gm` or the replacement"
                                   " GraphModule `replacement`")

    gm.graph.lint()

@compatibility(is_backward_compatible=True)
def replace_pattern(
        gm: GraphModule,
        pattern: Callable,
        replacement: Callable,
        is_match_filters: Optional[List[Callable]] = None) -> List[Match]:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

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
    # Get the module and graph for `gm`, `pattern`, `replacement`
    original_module = gm
    original_graph = original_module.graph
    pattern_module = symbolic_trace(pattern)
    pattern_graph = pattern_module.graph
    replacement_module = symbolic_trace(replacement)
    replacement_graph = replacement_module.graph

    # Find all possible pattern matches in original_graph. Note that
    # pattern matches may overlap with each other.
    matcher = _SubgraphMatcher(pattern_graph)
    matches: List[Match] = []

    # Consider each node as an "anchor" (deepest matching graph node)
    for anchor in original_graph.nodes:

        if matcher.matches_subgraph_from_anchor(anchor, original_module, pattern_module):

            def pattern_is_contained(nodes_map: Dict[Node, Node]) -> bool:
                # `lookup` represents all the nodes in `original_graph`
                # that are part of `pattern`
                lookup: Dict[Node, Node] = {v: k for k, v in nodes_map.items()}
                for n in lookup.keys():

                    # Nodes that can "leak"...
                    if not isinstance(lookup[n], Node):
                        continue
                    # Placeholders (by definition)
                    if lookup[n].op == "placeholder":
                        continue
                    # Pattern output (acts as a container)
                    if lookup[n].op == "output":
                        continue
                    # Result contained by pattern output (what we'll
                    # hook in to the new Graph, thus what we'll
                    # potentially use in other areas of the Graph as
                    # an input Node)
                    if (len(lookup[n].users) == 1
                            and list(lookup[n].users.keys())[0].op == "output"):
                        continue

                    if not isinstance(n, Node):
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
                                     nodes_map=copy.copy({
                                         key: value
                                         for key, value in matcher.nodes_map.items()
                                     })))

    # The set of all nodes in `original_graph` that we've seen thus far
    # as part of a pattern match
    replaced_nodes: Set[Node] = set()
    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = dict()

    # Return True if one of the nodes in the current match has already
    # been used as part of another match
    def overlaps_with_prev_match(match: Match) -> bool:
        for pn, gn in match.nodes_map.items():
            if not isinstance(pn, Node):
                continue
            if pn.op in ["placeholder", "output"]:
                continue
            if not isinstance(gn, Node):
                continue
            if gn in replaced_nodes and gn.op != "placeholder":
                return True
        return False

    if is_match_filters is None:
        is_match_filters = []

    def is_match(match: Match):
        # for mypy
        assert is_match_filters is not None
        for filter in is_match_filters:
            if not filter(match, pattern_graph, replacement_graph):
                return False
        return True

    for match in matches:
        # Skip overlapping matches
        if overlaps_with_prev_match(match):
            continue

        if not is_match(match):
            continue

        # Map replacement graph nodes to their copy in `original_graph`
        val_map: Dict[Node, Node] = {}

        pattern_placeholders = [n for n in pattern_graph.nodes
                                if n.op == "placeholder"]
        assert len(pattern_placeholders) > 0
        replacement_placeholders = [n for n in replacement_graph.nodes
                                    if n.op == "placeholder"]
        assert len(pattern_placeholders) == len(replacement_placeholders)
        placeholder_map = {r: p for r, p
                           in zip(replacement_placeholders, pattern_placeholders)}

        # node from `original_graph` that matched with the output node
        # in `pattern`
        subgraph_output: Node = match.anchor

        def mark_node_as_replaced(n: Node) -> None:
            if n not in match.nodes_map.values():
                return
            for n_ in n.all_input_nodes:
                mark_node_as_replaced(n_)
            replaced_nodes.add(n)

        for input_node in subgraph_output.all_input_nodes:
            mark_node_as_replaced(input_node)

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        for replacement_node in replacement_placeholders:
            # Get the `original_graph` placeholder node
            # corresponding to the current `replacement_node`
            pattern_node = placeholder_map[replacement_node]
            original_graph_node = match_changed_node.get(match.nodes_map[pattern_node], match.nodes_map[pattern_node])

            # Populate `val_map`
            val_map[replacement_node] = original_graph_node

        # Copy the replacement graph over
        with original_graph.inserting_before(subgraph_output):
            copied_output = original_graph.graph_copy(replacement_graph,
                                                      val_map)

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location

        # CASE 1: We need to hook the replacement subgraph in somewhere
        # in the middle of the graph. We replace the Node in the
        # original graph that corresponds to the end of the pattern
        # subgraph
        if subgraph_output.op != "output":
            pattern_outputs = [n for n in pattern_graph.nodes
                               if n.op == "output"]
            assert len(pattern_outputs) > 0
            replacement_outputs = [n for n in replacement_graph.nodes
                                   if n.op == "output"]
            assert len(replacement_outputs) == len(pattern_outputs)
            outputs_map = {p: r for r, p
                           in zip(replacement_outputs, pattern_outputs)}

            for pn, gn in match.nodes_map.items():
                if not isinstance(gn, Node):
                    continue
                if gn.op == "placeholder":
                    continue

                # Search for the node corresponding to the output of the pattern
                if pn.op != "output":
                    continue
                assert subgraph_output == gn

                # Update all anchor inputs to the new nodes
                rn = outputs_map[pn]
                for pn_input, rn_input in zip(pn.args, rn.args):
                    gn_input = match.nodes_map[pn_input]  # type: ignore[index]
                    rn_input_in_original_graph = val_map[rn_input]
                    gn_input.replace_all_uses_with(rn_input_in_original_graph)
                    # We store the updated node point in case other nodes want to use it
                    match_changed_node[gn_input] = rn_input_in_original_graph

            assert subgraph_output.op != "output"
        # CASE 2: The pattern subgraph match extends to the end of the
        # original graph, so we need to change the current graph's
        # output Node to reflect the insertion of the replacement graph.
        # We'll keep the current output Node, but update its args and
        # `_input_nodes` as necessary
        else:
            subgraph_output.args = ((copied_output,))
            if isinstance(copied_output, Node):
                subgraph_output._input_nodes = {copied_output: None}

        assert isinstance(copied_output, Node)
        # Erase the `pattern` nodes
        for node in reversed(original_graph.nodes):
            if len(node.users) == 0 and node.op != "output" and node.op != "placeholder":
                original_graph.erase_node(node)

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_submodules(gm, replacement)

    return matches
