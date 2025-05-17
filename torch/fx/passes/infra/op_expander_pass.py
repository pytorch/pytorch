from typing import Callable, Generic, Optional, TypeVar

import torch


T = TypeVar("T")

__all__ = ["OpExpanderPass"]


class OpExpanderPass(Generic[T]):
    def __init__(
        self,
        pattern_matcher: Callable[[torch.fx.Node], tuple[bool, Optional[T]]],
        node_expander: Callable[
            [torch.fx.Node, Optional[T], torch.fx.GraphModule], torch.fx.Node
        ],
        extra_filter: Optional[Callable[[torch.fx.Node], bool]] = None,
    ):
        """
        Initialize the OpExpander pass.

        Args:
            pattern_matcher: Returns (matched: bool, data: Optional[T])
            node_expander: Function to expand a node using optional match data of type T
            extra_filter: Optional extra filtering criteria for matching nodes
        """
        self.pattern_matcher = pattern_matcher
        self.node_expander = node_expander
        self.extra_filter = extra_filter

    def run(self, gm: torch.fx.GraphModule) -> bool:
        """
        Run the op expansion pass on the given graph module.

        Args:
            gm: The graph module to transform

        Returns:
            bool: True if any changes were made to the graph, False otherwise
        """
        matching_nodes = []

        for node in gm.graph.nodes:
            matches, match_data = self.pattern_matcher(node)
            if matches and (not self.extra_filter or self.extra_filter(node)):
                matching_nodes.append((node, match_data))

        changed = False
        for node, match_result in matching_nodes:
            expanded_node = self.node_expander(node, match_result, gm)
            if expanded_node is None:
                continue

            node.replace_all_uses_with(expanded_node)

            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.eliminate_dead_code()

        return changed
