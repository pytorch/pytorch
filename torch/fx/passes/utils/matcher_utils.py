from collections import defaultdict
import copy
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility

from torch.fx.subgraph_rewriter import Match

from typing import Dict, List, Set

__all__ = ['SubgraphMatcher']

@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    def __init__(self, pattern: Graph,
                 match_output: bool = False,
                 match_placeholder: bool = False,
                 fully_contained: bool = True,
                 remove_overlapping_matches: bool = True) -> None:
        """
        Args:
            pattern: the targeted matching pattern, represented in fx.Graph
            match_output: If True, output node in the pattern graph will be treated as a part of the targeted pattern.
                If False, output node is ignored during match.
            match_placeholder: If True, placeholder node in the pattern graph will be treated as a part of
                the targeted pattern. If False, placeholder nodes will be used a wildcard.
            fully_contained: If True, nodes (except placeholder and output_link_nodes ) can only be consumed
                by nodes within th pattern.
            remove_overlapping_matches: If True, in the case of overlapping matches, only the first match
                will be returned.
        """

        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.fully_contained = fully_contained
        self.remove_overlapping_matches = remove_overlapping_matches

        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an empty pattern")

        for node in pattern.nodes:
            if node.op != "output":
                assert len(node.users) > 0, \
                       "SubgraphMatcher cannot be initialized with an pattern with dead code"

        # TODO: assert pattern is a connected graph

        self.pattern_placeholder_nodes = [n for n in pattern.nodes if n.op == "placeholder"]
        output_node = next(iter(reversed(pattern.nodes)))
        # nodes returned by outputs
        self.pattern_returning_nodes: List[Node] = output_node.all_input_nodes

        self.pattern_anchors: List[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # If a node has output_node as the ONLY user, then this node is a graph sink,
            # and should be matched against as an anchor
            self.pattern_anchors = [n for n in output_node.all_input_nodes if len(n.users) == 1]

    def _nodes_are_equal(self, pn: Node, gn: Node) -> bool:
        # TODO: match args and kwargs

        # if exact match for placeholder is not required, then use placeholder as a wildcard
        if not self.match_placeholder and pn.op == "placeholder":
            return True

        if pn.op == gn.op:
            if pn.op == "placeholder" or pn.op == "output":
                return True
            return pn.target == gn.target
        return False

    def _is_contained(self, nodes_map: Dict[Node, Node]) -> bool:
        # `lookup` represents all the nodes in `original_graph`
        # that are part of `pattern`
        lookup: Dict[Node, Node] = {gn : pn for pn, gn in nodes_map.items()}
        for gn, pn in lookup.items():
            # Placeholders can be used by other nodes in the graphs
            if pn.op == "placeholder":
                continue

            # nodes returned by output are allowed to be used in other areas of the graph
            if pn in self.pattern_returning_nodes:
                continue

            for user in gn.users:
                # If this node has users that were not in `lookup`, then it must leak out of the
                # pattern subgraph
                if user not in lookup:
                    return False
        return True

    def _remove_overlapping_matches(self, matches: List[Match]) -> List[Match]:
        non_overlapping_matches: List[Match] = list()
        nodes_matched: Set[Node] = set()

        for match in matches:
            found_overlap = False
            for pn, gn in match.nodes_map.items():
                if pn.op not in {"placeholder", "output"} and gn in nodes_matched:
                    found_overlap = True
                    break

            if not found_overlap:
                non_overlapping_matches.append(match)
                for pn, gn in match.nodes_map.items():
                    if pn.op not in {"placeholder", "output"}:
                        nodes_matched.add(gn)
        return non_overlapping_matches

    def _match_nodes(self, pn: Node, gn: Node, match: Match) -> bool:

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficienty way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        if not self._nodes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        match.nodes_map[pn] = gn

        if pn.op == "placeholder":
            return True

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = (len(pn.all_input_nodes) == len(gn.all_input_nodes) and
                       all(self._match_nodes(pn_, gn_, match) for pn_, gn_
                           in zip(pn.all_input_nodes, gn.all_input_nodes)))

        if not match_found:
            match.nodes_map.pop(pn)
            return False

        return True

    def match(self, graph: Graph) -> List[Match]:
        """
        Subgraph pattern matcher is implemented with the backtracking style in the following steps:

        1. We first identify all the anchor nodes in the pattern graph. The anchor nodes
        are the "sinks" (nodes with no user other than the output node) of the pattern graph.
        One pattern graph could have multiple anchors if it has multiple return values.

        2. In the target graph, we identify the potential candidate nodes that can be matched
        with each anchor. These anchor-candidate pairs are the starting points for
        pairwise per-node matching.

        3. For each anchor-candidate pair, we simultaneously traverse backwards (DFS) in both
        pattern and target graphs. For every pattern nodes along traversal path, we compare it
        against the target nodes. In case any comparison failed, the match for this anchor-candidate
        pair fails. A match is found when DFS completes traversing the graph. See `self._match_nodes`
        for more details.

        4. In the case of multiple anchors, every anchor will need to find a match using step 3.
        In addition, the matches found between anchors need to have a common intersection node
        in order for the match to be valid. This is implemented with backtracking. See `backtracking`
        for more details.

        Notice: graph traversal must be done in the reverser order because a tensor can have multiple
        consumers, but can only have a single producer. Only with reverser order, we can we jointly
        traverse the pattern and target graph in a deterministic path.
        """

        # find candidate nodes to match with pattern anchors
        match_candidates: Dict[Node, List[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if self._nodes_are_equal(pattern_anchor, node):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())
        matches: List[Match] = []

        def backtracking(anchor_index, match):
            if anchor_index == len(match_candidates_list):
                match.placeholder_nodes = [match.nodes_map[pn] for pn in self.pattern_placeholder_nodes]
                match.returning_nodes = [match.nodes_map[pn] for pn in self.pattern_returning_nodes]
                matches.append(match)
                return

            pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
            saved_match = copy.copy(match)

            for node in candidate_nodes:
                match_found = self._match_nodes(pattern_anchor, node, match)
                if match_found:
                    # match next anchor
                    backtracking(anchor_index + 1, match)

                    # revert to saved_match before matching with current anchor
                    match = copy.copy(saved_match)

        # TODO: Match's anchor is set to the first of self.pattern_anchors[0] for now,
        # need to update the sematics of this field
        match = Match(anchor=self.pattern_anchors[0])
        backtracking(0, match)

        if self.fully_contained:
            matches = [match for match in matches if self._is_contained(match.nodes_map)]

        if self.remove_overlapping_matches:
            matches = self._remove_overlapping_matches(matches)

        return matches
