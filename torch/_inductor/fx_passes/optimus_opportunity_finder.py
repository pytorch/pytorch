import logging
from collections import Counter

from .group_batch_fusion import find_independent_subset_greedy
log = logging.getLogger(__name__)


def is_valid_node_to_optimize(node):
    blacklist = ["getitem", "split", "squeeze", "permute"]
    return not any(i in str(node.target) for i in blacklist)

def optimus_opportunity_finder_passes(graph):
    log.info("=====================================================")
    log.info("Optimus Opportunity Finder start to analyze fx graph.")
    targets = [
        node.target
        for node in graph.nodes
        if is_valid_node_to_optimize(node)
        and (node.op == "call_function" or node.op == "call_method")
    ]
    item_counter = Counter(targets)
    log.info("Optimus Opportunity Finder found %d call_function and nodes.", len(item_counter))
    for target in item_counter:
        log.info("Analysis for %s. Find %d in the graph", target, item_counter[target])

        candidate_nodes = [node for node in graph.nodes if node.target == target]
        for subset in find_independent_subset_greedy(candidate_nodes):
            log.info(
               "Find horizontal fusion opportunies. Can fuse %d", len(subset)
            )
