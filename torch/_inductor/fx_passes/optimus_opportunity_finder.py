import logging
from collections import Counter, defaultdict

import torch
from torch.fx.experimental.symbolic_shapes import free_symbols

from .group_batch_fusion import (
    find_independent_subset_greedy,
    graph_search_options,
    is_node_meta_valid,
)


log = logging.getLogger(__name__)


def is_valid_node_to_optimize(node: torch.fx.Node) -> bool:
    target_list = ["aten.bmm", "aten.mm", "aten.addmm"]
    return any(i in str(node.target) for i in target_list)


def optimus_opportunity_finder_passes(
    graph: torch.fx.Graph, pre_grad: bool = False
) -> None:
    log.debug("=====================================================")
    log.debug("Optimus Opportunity Finder start to analyze fx graph.")
    targets = [
        node.target
        for node in graph.nodes
        if is_valid_node_to_optimize(node)
        and (node.op == "call_function" or node.op == "call_method")
    ]
    item_counter = Counter(targets)
    log.debug(
        "Optimus Opportunity Finder found %s call_function and nodes.",
        len(item_counter),
    )
    keyword = "example_value" if pre_grad else "val"
    for target in item_counter:
        log.debug(
            "Analysis for %s. Found %s in the graph.", target, item_counter[target]
        )
        candidate_nodes = [node for node in graph.nodes if node.target == target]
        for subset in find_independent_subset_greedy(
            candidate_nodes, graph_search_options
        ):
            subset_nodes_shape_counter: dict[torch.Size, int] = defaultdict(int)
            for node in subset:
                if not is_node_meta_valid(node):
                    log.debug("example value absent for node: %s:", node)
                    continue
                if not isinstance(node.meta[keyword], torch.Tensor):
                    log.debug("example value is not a tensor for node: %s", node)
                    continue
                if free_symbols(node.meta[keyword].shape):
                    log.debug("dynamic shape not supported for node: %s", node)
                    continue
                subset_nodes_shape_counter[node.meta[keyword].shape] += 1
            log.debug(
                "Find horizontal fusion opportunies. Can fuse %s node shapes.",
                len(subset_nodes_shape_counter),
            )
            for shape, count in sorted(
                subset_nodes_shape_counter.items(), key=lambda x: x[1], reverse=True
            ):
                log.debug("Shape: %s, Count: %s", shape, count)
