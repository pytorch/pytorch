from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .passes.utils.matcher_with_name_node_map_utils import InternalMatch

__all__ = ['Match', 'replace_pattern', 'replace_pattern_with_filters', "ReplacedPatterns"]

@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]

@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: List[Node]

def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_attr(gm: torch.nn.Module, target: str) -> Optional[Any]:
        module_path, _, attr_name = target.rpartition(".")
        try:
            mod: torch.nn.Module = gm.get_submodule(module_path)
        except AttributeError:
            return None
        attr = getattr(mod, attr_name, None)
        return attr

    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":

            gm_attr = try_get_attr(gm, node.target)
            replacement_attr = try_get_attr(replacement, node.target)

            # CASE 1: This target already exists as an attribute in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_attr is not None:
                continue

            # CASE 2: The target exists as an attribute in `replacement`
            # only, so we need to copy it over.
            elif replacement_attr is not None:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)

            # CASE 3: The target doesn't exist as an attribute in `gm`
            # or `replacement`
            else:
                raise RuntimeError('Attempted to create a "', node.op,
                                   '" node during subgraph rewriting '
                                   f"with target {node.target}, but "
                                   "the referenced attribute does not "
                                   "exist in the replacement GraphModule")

    gm.graph.lint()


@compatibility(is_backward_compatible=True)
def replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule]
) -> List[Match]:
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
            def __init__(self) -> None:
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
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    return [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]


# Experimental API, not backward compatible
@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[List[Callable[["InternalMatch", Graph, Graph], bool]]] = None,
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:
    """
    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.

    Args:
        ``match_filters``: A list of functions that take in
            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating
            whether the match satisfies the condition.
            See matcher_utils.py for definition of InternalMatch.
    """

    return _replace_pattern(gm, pattern, replacement, match_filters, ignore_literals)


def _replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[List[Callable[["InternalMatch", Graph, Graph], bool]]] = None,
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:

    from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch

    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph

    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        replacement_graph = replacement
    else:
        replacement_graph = symbolic_trace(replacement).graph

    matcher = SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True, ignore_literals=ignore_literals)
    _matches: List[InternalMatch] = matcher.match(original_graph)

    # Filter out matches that don't match the filter
    _matches = [
        m for m in _matches
        if all(match_filter(m, original_graph, pattern_graph)
               for match_filter in match_filters)
    ]

    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == "placeholder"]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    # Update match.placeholder_nodes and match.nodes_map with the node that replaced gn
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[list(match.nodes_map.values()).index(gn)]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            user_nodes.update(n.users)
        assert user_nodes, "The returning_nodes should have at least one user node"

        if len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        with original_graph.inserting_before(first_user_node):  # type: ignore[possibly-undefined]
            copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes, )

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes: List[Node] = [v for v in val_map.values() if v not in match.placeholder_nodes]

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)  # type: ignore[arg-type]
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):  # type: ignore[arg-type]
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return match_and_replacements
