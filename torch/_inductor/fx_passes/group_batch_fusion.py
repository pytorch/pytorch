import collections
import logging
import operator

import torch
from torch._dynamo.utils import counters

from .. import config
from ..pattern_matcher import CallFunctionVarArgs

try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False
    pass

aten = torch.ops.aten

log = logging.getLogger(__name__)

maximum_group_size = 50


def _has_path(src_node, dest_node, cache):
    """
    If the graph has a path from `src_node` to `dest_node`, return True
    """
    cached = cache[id(src_node)][id(dest_node)]
    if cached is not None:
        return cached
    src_input_nodes = src_node.all_input_nodes
    if dest_node in src_input_nodes:
        cache[id(src_node)][id(dest_node)] = True
        return True

    for x in src_input_nodes:
        if _has_path(x, dest_node, cache):
            cache[id(src_node)][id(dest_node)] = True
            return True

    cache[id(src_node)][id(dest_node)] = False
    return False


def _get_independent_node_subsets(node_list):
    """
    Return an iterator of node subset, each subset only contains nodes
    those are independent with each other.
    """
    cache = collections.defaultdict(lambda: collections.defaultdict(lambda: None))

    while len(node_list) > 0:
        next_round = []

        independent_set = []

        def is_dependent(node_set, other):
            for node in node_set:
                if _has_path(node, other, cache) or _has_path(other, node, cache):
                    return True

            return False

        for node in node_list:
            if len(independent_set) < maximum_group_size and not is_dependent(
                independent_set, node
            ):
                independent_set.append(node)
            else:
                next_round.append(node)

        yield independent_set

        cache.clear()

        node_list = next_round


class GroupBatchFusionBase:
    def match(self, node):
        raise NotImplementedError("match called on base")

    def fuse(self, graph, subset):
        raise NotImplementedError("fuse called on base")


class GroupFusion(GroupBatchFusionBase):
    """
    Fuse ops in a group way, e.g, fuse mm/addmm of arbitrary input shapes with fbgemm.gmm.
    """

    pass


class BatchFusion(GroupBatchFusionBase):
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """

    pass


class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node):
        input_shape = node.args[1].meta["tensor_meta"].shape
        weight_shape = node.args[2].meta["tensor_meta"].shape
        return (
            node.kwargs.get("beta", 1.0) == 1.0
            and node.kwargs.get("alpha", 1.0) == 1.0
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
        )

    def match(self, node):
        if CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ("group_linear", bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph, subset):
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args

            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        if all(bias is None for bias in group_biases):
            group_biases = None

        with graph.inserting_before(subset[0]):
            fused_mm = graph.call_function(
                torch.ops.fbgemm.gmm,
                args=(group_inputs, group_weights, group_biases),
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)


def apply_group_batch_fusion(graph, rule):
    fusible_groups = collections.defaultdict(list)

    for node in graph.nodes:
        group_key = rule.match(node)
        if group_key:
            fusible_groups[group_key].append(node)

    log.debug("Generated fusible groups: %s", fusible_groups)

    for fusible_nodes in fusible_groups.values():
        subset_list = list(_get_independent_node_subsets(fusible_nodes))
        for subset in subset_list:
            if len(subset) <= 1:
                continue

            rule.fuse(graph, subset)

            if isinstance(rule, GroupFusion):
                counters["inductor"]["group_fusion"] += 1
            else:
                counters["inductor"]["batch_fusion"] += 1


def group_batch_fusion_passes(graph: torch.fx.Graph):
    fusions = []

    if config.group_fusion and has_fbgemm:
        fusions += [GroupLinearFusion()]

    for rule in fusions:
        apply_group_batch_fusion(graph, rule)
