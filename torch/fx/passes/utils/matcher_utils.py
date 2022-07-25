from collections import defaultdict
import copy
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility

from torch.fx.subgraph_rewriter import Match

from typing import Dict, List


@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    def __init__(self, pattern: Graph,
                 match_output: bool = False,
                 match_placeholder: bool = False) -> None:
        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder

        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an empty pattern")

        for node in pattern.nodes:
            if node.op != "output":
                assert len(node.users) > 0, \
                       "SubgraphMatcher cannot be initialized with an pattern with dead code"

        # TODO: assert pattern is a connected graph

        output_node = next(iter(reversed(pattern.nodes)))

        self.pattern_anchors: List[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # If a node has output_node as the ONLY user, then this node is a graph sink,
            # and should be matched against as an anchor
            self.pattern_anchors = [n for n in output_node.all_input_nodes if len(n.users) == 1]

    @staticmethod
    def _nodes_are_equal(pn: Node, gn: Node) -> bool:
        # TODO: match args and kwargs
        if pn.op == gn.op:
            if pn.op == "placeholder" or pn.op == "output":
                return True
            return pn.target == gn.target
        return False

    def _match_nodes(self, pn: Node, gn: Node, match: Match) -> bool:

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficienty way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        # skip matching placeholder if match_placeholder is False
        if not self.match_placeholder and pn.op == "placeholder":
            return True

        if not SubgraphMatcher._nodes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        match.nodes_map[pn] = gn

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = (len(pn.all_input_nodes) == len(gn.all_input_nodes)
                        and all(self._match_nodes(pn_, gn_, match) for pn_, gn_
                                in zip(pn.all_input_nodes, gn.all_input_nodes)))

        if not match_found:
            match.nodes_map.pop(pn)
            return False

        return True

    def match(self, graph: Graph) -> List[Match]:
        # find candidate nodes to match with pattern anchors
        match_candidates: Dict[Node, List[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if SubgraphMatcher._nodes_are_equal(pattern_anchor, node):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())

        matches: List[Match] = []
        def backtracking(anchor_index, match):
            if anchor_index == len(match_candidates_list):
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

        match = Match(anchor=None, nodes_map={})
        backtracking(0, match)

        return matches
