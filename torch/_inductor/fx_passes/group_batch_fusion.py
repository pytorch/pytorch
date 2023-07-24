import collections
import logging
import operator

import torch
from torch._dynamo.utils import counters

from .. import config
from ..pattern_matcher import CallFunctionVarArgs, get_arg_value

try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False
    pass

aten = torch.ops.aten

log = logging.getLogger(__name__)

maximum_group_size = 150


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


class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    def is_node_meta_valid(self, node):
        if node is None:
            return True
        else:
            if "example_value" in node.meta:
                return True
            else:
                return False

    def match(self, node):
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 2, "weight")
            bias = get_arg_value(node, 3, "bias")
            group_key = (
                (
                    "batch_layernorm",
                    str(input.meta["example_value"].shape),
                    str(weight.meta["example_value"].shape)
                    if weight is not None
                    else "",
                    str(bias.meta["example_value"].shape) if bias is not None else "",
                    str(get_arg_value(node, 1, "normalized_shape")),
                    str(get_arg_value(node, 4, "eps")),
                )
                if "example_value" in input.meta
                and self.is_node_meta_valid(weight)
                and self.is_node_meta_valid(bias)
                else None
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph, subset):
        group_inputs = []
        group_shapes = []
        group_weights = []
        group_biases = []
        group_epss = []
        group_nodes = []
        for node in subset:
            group_nodes.append(node)
            group_inputs.append(get_arg_value(node, 0, "input"))
            group_shapes.append(get_arg_value(node, 1, "normalized_shape"))
            group_weights.append(get_arg_value(node, 2, "weight"))
            group_biases.append(get_arg_value(node, 3, "bias"))
            eps = get_arg_value(node, 4, "eps")
            if eps is None:
                eps = 1e-5
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])

        if all(bias is None for bias in group_biases):
            group_biases = None
        if all(weight is None for weight in group_weights):
            group_weights = None
        assert all(
            eps == group_epss[0] for eps in group_epss
        ), "all epsilon values must be equal"

        with graph.inserting_before(subset[0]):
            stack_input = graph.call_function(
                torch.stack, args=(group_inputs, stack_dim)
            )
            if group_weights is not None:
                stack_weight = graph.call_function(torch.stack, args=(group_weights,))
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(torch.stack, args=(group_biases,))
            else:
                stack_bias = None

            batch_layer_norm = graph.call_function(
                torch.nn.functional.layer_norm,
                args=(stack_input, group_shapes[-1]),
                kwargs={"eps": group_epss[-1]},
            )

            if group_weights is not None and group_biases is not None:
                batch_layer_norm = graph.call_function(
                    torch.addcmul, args=(stack_bias, stack_weight, batch_layer_norm)
                )
            elif group_weights is not None and group_biases is None:
                batch_layer_norm = graph.call_function(
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
            elif group_weights is None and group_biases is not None:
                batch_layer_norm = graph.call_function(
                    torch.add, args=(stack_bias, batch_layer_norm)
                )

            batch_layer_norm_unbind = graph.call_function(
                torch.unbind,
                args=(batch_layer_norm,),
                kwargs={"dim": stack_dim},
            )

        for i, node in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):
                new_node = graph.call_function(
                    operator.getitem, args=(batch_layer_norm_unbind, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)


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


def group_batch_fusion_post_grad_passes(graph: torch.fx.Graph):
    fusions = []
    if config.group_fusion and has_fbgemm:
        fusions += [GroupLinearFusion()]

    for rule in fusions:
        apply_group_batch_fusion(graph, rule)


def group_batch_fusion_pre_grad_passes(graph: torch.fx.Graph):
    fusions = []

    if config.batch_fusion and has_fbgemm:
        fusions += [BatchLayernormFusion()]

    for rule in fusions:
        apply_group_batch_fusion(graph, rule)
