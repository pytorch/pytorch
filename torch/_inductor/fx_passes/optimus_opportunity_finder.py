import logging
from collections import Counter, defaultdict
from typing import Dict

import torch

from .group_batch_fusion import find_independent_subset_greedy, graph_search_options

log = logging.getLogger(__name__)


def is_valid_node_to_optimize(node):
    blacklist = ["getitem", "split", "squeeze", "permute"]
    return not any(i in str(node.target) for i in blacklist)


def optimus_opportunity_finder_passes(graph, pre_grad=True):
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
        f"Optimus Opportunity Finder found {len(item_counter)} call_function and nodes.",
    )
    keyword = "example_value" if pre_grad else "tensor_meta"
    for target in item_counter:
        log.debug(f"Analysis for {target}. Find {item_counter[target]} in the graph.")
        candidate_nodes = [node for node in graph.nodes if node.target == target]
        subset_nodes_shape_counter: Dict[torch.Size, int] = defaultdict(int)
        for subset in find_independent_subset_greedy(
            candidate_nodes, graph_search_options
        ):
            for node in subset:
                if keyword not in node.meta:
                    log.debug(f"example value absent for node: {node}")
                    continue
                subset_nodes_shape_counter[node.meta[keyword].shape] += 1
            log.debug(
                f"Find horizontal fusion opportunies. Can fuse {len(subset_nodes_shape_counter)}."
            )
            log.debug(f"The shapes statistics dict is: {subset_nodes_shape_counter}")
