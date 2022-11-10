from dataclasses import dataclass, field
from collections import defaultdict
import copy
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Callable, Dict, List, Set, Any
import logging
import os

__all__ = ['SubgraphMatcher', 'InternalMatch']

# Set`PYTORCH_MATCHER_LOGLEVEL=INFO` to see debug logs
def _init_logger():
    logger = logging.getLogger(__name__)

    level = os.environ.get('PYTORCH_MATCHER_LOGLEVEL', 'WARNING').upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger

logger = _init_logger()

@compatibility(is_backward_compatible=False)
@dataclass
class InternalMatch():
    # Nodes from which the match was found
    anchors: List[Node]
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node] = field(default_factory=dict)

    # nodes in target graph that are matched placeholder in pattern
    placeholder_nodes: List[Node] = field(default_factory=list)

    # nodes in matched subgraph returned by output
    returning_nodes: List[Node] = field(default_factory=list)

    def __copy__(self):
        return InternalMatch(anchors=self.anchors, nodes_map=self.nodes_map.copy(),
                             placeholder_nodes=self.placeholder_nodes.copy(),
                             returning_nodes=self.returning_nodes.copy())

@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    def __init__(self, pattern: Graph,
                 match_output: bool = False,
                 match_placeholder: bool = False,
                 remove_overlapping_matches: bool = True,
                 custom_node_args_matcher: Callable[[Node, Node], bool] = None) -> None:
        """
        Args:
            pattern: the targeted matching pattern, represented in fx.Graph.
            match_output: If True, output node in the pattern graph will be treated as a part of the targeted pattern.
                If False, output node is ignored during match.
            match_placeholder: If True, placeholder node in the pattern graph will be treated as a part of
                the targeted pattern. If False, placeholder nodes will be used a wildcard.
            remove_overlapping_matches: If True, in the case of overlapping matches, only the first match
                will be returned.
            custom_node_args_matcher: If present, will be called to match args that are not from upstream nodes.
                Example of such args include kernel size, stride, etc. Returns True if match is successful in which
                case exact the args match will be skipped. Returns False if it's not a match or if the node type is
                not relevant to the matcher, in which the exact args match will be performed as usual.
        """

        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.remove_overlapping_matches = remove_overlapping_matches
        self.custom_node_args_matcher = custom_node_args_matcher

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

        # Placeholders can be used by other nodes in the graphs
        lookup: Dict[Node, Node] = {gn : pn for pn, gn in nodes_map.items() if pn.op != "placeholder"}

        for gn, pn in lookup.items():
            # nodes returned by output are allowed to be used in other areas of the graph
            if pn in self.pattern_returning_nodes:
                continue

            for user in gn.users:
                # If this node has users that were not in `lookup`, then it must leak out of the
                # pattern subgraph
                if user not in lookup:
                    return False
        return True

    def _remove_overlapping_matches(self, matches: List[InternalMatch]) -> List[InternalMatch]:
        non_overlapping_matches: List[InternalMatch] = list()
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

    def _match_args(self, pn: Any, gn: Any, match: InternalMatch) -> bool:
        assert not (isinstance(pn, Node) and isinstance(gn, Node)), "pn and gn cannot both be Node"

        if isinstance(pn, Node) and not isinstance(gn, Node):
            if pn.op == "placeholder":
                # Check if we've already matched these nodes in the current
                # traversal
                if pn in match.nodes_map:
                    return match.nodes_map[pn] == gn

                match.nodes_map[pn] = gn
                return True
            else:
                return False
        elif not isinstance(pn, Node) and isinstance(gn, Node):
            return False
        else:
            return type(gn) == type(pn) and gn == pn

    def _match_nodes(self, pn: Node, gn: Node, match: InternalMatch) -> bool:
        logger.info(f"  matching {pn} to {gn}")

        assert isinstance(pn, Node) and isinstance(gn, Node), str(f"pn and gn must be Node, pn: {pn}, gn: {gn}")

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficienty way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        if not self._nodes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`, and save a local copy of match
        saved_match = copy.copy(match)
        match.nodes_map[pn] = gn

        if pn.op == "placeholder":
            return True

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = True

        def flatten_args(args) -> List[Any]:
            # Recursively flatten args
            result : List[Any] = []
            for arg in args:
                # flatten the list, if only it's a list/tuple of nodes
                if isinstance(arg, (list, tuple)) and len(arg) > 0 and isinstance(arg[0], Node):
                    result.extend(flatten_args(arg))
                else:
                    result.append(arg)

            return result

        pn_flatten_args = flatten_args(pn.args)
        gn_flatten_args = flatten_args(gn.args)

        if pn.kwargs.keys() == gn.kwargs.keys():
            for key in pn.kwargs.keys():
                pn_flatten_args.append(pn.kwargs[key])
                gn_flatten_args.append(gn.kwargs[key])
        else:
            match_found = False

        if self.custom_node_args_matcher:
            custom_args_match = self.custom_node_args_matcher(pn, gn)
        else:
            custom_args_match = False

        if match_found and len(pn_flatten_args) == len(gn_flatten_args):
            for pn_, gn_ in zip(pn_flatten_args, gn_flatten_args):
                if isinstance(gn_, Node) and isinstance(pn_, Node):
                    matched = self._match_nodes(pn_, gn_, match)
                elif (
                    isinstance(gn_, Node)
                    or isinstance(pn_, Node)
                    or not custom_args_match
                ):
                    matched = self._match_args(pn_, gn_, match)
                else:
                    matched = custom_args_match

                if not matched:
                    match_found = False
                    break
        else:
            match_found = False

        if not match_found:
            # revert to saved_match before matching with current node
            match = copy.copy(saved_match)
            return False

        return True

    def match(self, graph: Graph) -> List[InternalMatch]:
        """
        Returns:
            The matched subgraphs.
            Thre returned subgraph would be fully self-contained, meaning the nodes (except placeholder
            and nodes returned by output) can only be consumed by nodes within the matched subgraph.

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

        Warning: In theory, this backtracking algorithm have an **exponential** time complexity. However,
        in practice, it's unlikely to blow up.

        """
        from torch.fx.passes.utils.fuser_utils import validate_partition

        # find candidate nodes to match with pattern anchors
        match_candidates: Dict[Node, List[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if self._nodes_are_equal(pattern_anchor, node):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())

        logger.info(f"Initial match_candidates_list: {match_candidates_list}\n")

        matches: List[InternalMatch] = []

        def backtracking(anchor_index, match):
            if anchor_index == len(match_candidates_list):
                match.placeholder_nodes = [match.nodes_map[pn] for pn in self.pattern_placeholder_nodes]
                match.returning_nodes = [match.nodes_map[pn] for pn in self.pattern_returning_nodes]
                matches.append(match)

                logger.info(f"Found a match: {match}\n")
                return

            pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
            saved_match = copy.copy(match)

            for node in candidate_nodes:
                logger.info(f"Trying to match anchor {pattern_anchor} to {node}")

                match_found = self._match_nodes(pattern_anchor, node, match)
                if match_found:
                    # match next anchor
                    backtracking(anchor_index + 1, match)
                else:
                    logger.info(f"Failed to match anchor {pattern_anchor} to {node}\n")

                # revert to saved_match before matching with current anchor
                match = copy.copy(saved_match)

        match = InternalMatch(anchors=self.pattern_anchors)
        if match_candidates_list:
            backtracking(0, match)

        # filter out the matches where the subgraph is not fully_contained
        before = len(matches)
        matches = [match for match in matches if self._is_contained(match.nodes_map)]
        after = len(matches)
        if before != after:
            logger.info(f"Filtered out {before - after} matches because they are not fully contained")

        # filter out the matches that that forms a cycle if the subgraph is fused
        valid_matches = []
        for match in matches:
            matched_compute_nodes = \
                [gn for pn, gn in match.nodes_map.items() if pn.op not in {"placeholder", "output"}]
            if validate_partition(matched_compute_nodes):
                valid_matches.append(match)
        if len(valid_matches) != len(matches):
            logger.info(f"Filtered out {len(matches) - len(valid_matches)} matches because \
                          matched subgraph would form a cycle if fused")

        if self.remove_overlapping_matches:
            before = len(valid_matches)
            matches = self._remove_overlapping_matches(valid_matches)
            after = len(matches)
            if before != after:
                logger.info(f"Filtered out {before - after} matches because matched subgraphs are overlapping")

        logger.info(f"Matches returned: {matches}")

        return matches
