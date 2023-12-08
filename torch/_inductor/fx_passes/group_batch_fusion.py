import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph

from .. import config
from ..pattern_matcher import (
    CallFunctionVarArgs,
    get_arg_value,
    stable_topological_sort,
)

try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False
    pass

aten = torch.ops.aten

log = logging.getLogger(__name__)

MIN_FUSE_SET_SIZE = 5
MAX_FUSE_SET_SIZE = 300
MAX_FUSE_SEARCH_DEPTH = 5
# The maximum tensor size that can go into the fusion group
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR = 4096

# exclude these nodes from BFS
# excluding get item improves optimizer compilation time by 60s
SEARCH_EXCLUSIONS = {operator.getitem}


default_graph_search_options = {
    "min_fuse_set_size": MIN_FUSE_SET_SIZE,
    "max_fuse_set_size": MAX_FUSE_SET_SIZE,
    "max_fuse_search_depth": MAX_FUSE_SEARCH_DEPTH,
    "max_fuse_tensor_size_group_linear": MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR,
}

graph_search_options = default_graph_search_options


class GroupBatchFusionBase:
    def __init__(self, **kwargs):
        self.graph_search_options = kwargs.pop(
            "graph_search_options", default_graph_search_options
        )

    def match(self, node):
        raise NotImplementedError("match called on base")

    def fuse(self, graph, subset):
        raise NotImplementedError("fuse called on base")


PRE_GRAD_FUSIONS: Dict[str, GroupBatchFusionBase] = dict()
POST_GRAD_FUSIONS: Dict[str, GroupBatchFusionBase] = dict()


def register_fusion(name: str, pre_grad=True):
    def decorator(fusion_cls: GroupBatchFusionBase):
        if pre_grad:
            PRE_GRAD_FUSIONS[name] = fusion_cls
        else:
            POST_GRAD_FUSIONS[name] = fusion_cls
        return fusion_cls

    return decorator


def list_group_batch_fusions(pre_grad=True) -> List[str]:
    if pre_grad:
        return list(PRE_GRAD_FUSIONS.keys())
    else:
        return list(POST_GRAD_FUSIONS.keys())


def decompose_stack(graph: torch.fx.GraphModule, input_tensors: List[Any]) -> Any:
    unsqueezed_inputs = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(aten.unsqueeze, args=(input_tensor, 0))
        unsqueezed_inputs.append(unsqueezed_input)
    stacked_inputs = graph.call_function(
        aten.cat,
        args=(unsqueezed_inputs, 0),
    )
    return stacked_inputs


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


class BatchPointwiseOpsFusionFactory(BatchFusion):
    def __init__(self, op, **kwargs):
        super().__init__(**kwargs)
        self.op = op


@register_fusion("batch_linear_post_grad", pre_grad=False)
class PostGradBatchLinearFusion(BatchFusion):
    """
    Fuse ops in a batch way in post grad (aten level).
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        return (
            node.kwargs.get("beta", 1.0) == 1.0 and node.kwargs.get("alpha", 1.0) == 1.0
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        return len(input.meta["tensor_meta"].shape) == 2

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, int, int, int, bool]]:
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None

        # only handle the cases where inputs are 2D tensors
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):
            return None
        m, k = input_m.meta["tensor_meta"].shape
        n = weight_m.meta["tensor_meta"].shape[1]
        batch_key = ("batch_linear", m, k, n, bias_m is not None)
        return batch_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_nodes = []

        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            elif CallFunctionVarArgs(aten.mm.default).match(node):
                input, weight = node.args
                bias = None
            batch_nodes.append(node)
            batch_inputs.append(input)
            batch_weights.append(weight)
            batch_biases.append(bias)

        with graph.inserting_before(subset[-1]):
            fused_inputs = decompose_stack(graph, batch_inputs)
            fused_weights = decompose_stack(graph, batch_weights)
            fused_bmm = graph.call_function(
                aten.bmm,
                args=(fused_inputs, fused_weights),
            )

        for i, original_mm in enumerate(batch_nodes):
            has_bias = False
            with graph.inserting_after(fused_bmm):
                new_mm = graph.call_function(aten.select, args=((fused_bmm, 0, i)))
                if batch_biases[i]:
                    has_bias = True
                    new_bias_add = graph.call_function(
                        aten.add, args=((batch_biases[i], new_mm))
                    )
            new_mm_cont = new_bias_add if has_bias else new_mm
            original_mm.replace_all_uses_with(new_mm_cont)
            new_mm_cont.meta.update(original_mm.meta)
            graph.erase_node(original_mm)


@register_fusion("group_linear", pre_grad=False)
class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[1].meta["tensor_meta"].shape
        weight_shape = node.args[2].meta["tensor_meta"].shape
        return (
            node.kwargs.get("beta", 1.0) == 1.0
            and node.kwargs.get("alpha", 1.0) == 1.0
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[0].meta["tensor_meta"].shape
        weight_shape = node.args[1].meta["tensor_meta"].shape
        return (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool]]:
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._mm_node_can_be_fused(node):
            group_key = ("group_linear", True)
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ("group_linear", bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            else:
                assert CallFunctionVarArgs(aten.mm.default).match(node)
                input, weight = node.args
                bias = None

            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]
        group_biases: Optional[List[Any]]

        with graph.inserting_before(subset[0]):
            fused_mm = graph.call_function(
                torch.ops.fbgemm.gmm.default,
                args=(group_inputs, group_weights, group_biases),
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)


@register_fusion("batch_linear_lhs")
class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> Optional[Tuple[str, bool, Any]]:
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            bias = get_arg_value(node, 2, "bias")
            group_key = ("batch_linear_lhs", bias is None, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_input = None
        batch_weights = []
        batch_biases = []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                assert batch_input is input
            batch_weights.append(weight)
            if bias:
                batch_biases.append(bias)
            split_sections.append(weight.meta["example_value"].shape[0])

        with graph.inserting_before(subset[0]):
            cat_weights = graph.call_function(
                torch.cat, args=(batch_weights,), kwargs={"dim": 0}
            )
            transposed_weights = graph.call_function(
                torch.transpose, args=(cat_weights, 0, 1)
            )
            if len(batch_biases) > 0:
                cat_biases = graph.call_function(
                    torch.cat, args=(batch_biases,), kwargs={"dim": 0}
                )
                fused_lhs = graph.call_function(
                    torch.addmm,
                    args=(cat_biases, batch_input, transposed_weights),
                )
            else:
                fused_lhs = graph.call_function(
                    torch.mm,
                    args=(batch_input, transposed_weights),
                )
            fused_lhs_list = graph.call_function(
                torch.split, args=(fused_lhs, split_sections), kwargs={"dim": 1}
            )

        for i, node in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):
                new_node = graph.call_function(
                    operator.getitem, args=(fused_lhs_list, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)


def is_node_meta_valid(node: Optional[torch.fx.Node]):
    if node is None:
        return True
    if "example_value" not in node.meta:
        return False
    return True


def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    return (
        is_node_meta_valid(node)
        and is_node_meta_valid(input)
        and is_node_meta_valid(weight)
        and len(input.meta["example_value"].shape) == 2
        and len(weight.meta["example_value"].shape) == 2
    )


@register_fusion("batch_linear")
class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            group_key = (
                "batch_linear_pre_grad",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, "input"))
            batch_weights.append(get_arg_value(node, 1, "weight"))
            batch_biases.append(get_arg_value(node, 2, "bias"))

        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            stack_weights = graph.call_function(
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            transpose_weight = graph.call_function(
                torch.transpose, args=(stack_weights, 1, 2)
            )
            if all(bias is None for bias in batch_biases):
                bmm = graph.call_function(
                    torch.bmm,
                    args=(stack_inputs, transpose_weight),
                )
            else:
                stack_biases = graph.call_function(
                    torch.stack, args=(batch_biases,), kwargs={"dim": 0}
                )
                unsqueeze_biases = graph.call_function(
                    torch.unsqueeze, args=(stack_biases, 1)
                )
                bmm = graph.call_function(
                    torch.baddbmm,
                    args=(unsqueeze_biases, stack_inputs, transpose_weight),
                )

            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={"dim": 0})
            for i, linear in enumerate(batch_nodes):
                with graph.inserting_after(bmm):
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)


@register_fusion("batch_layernorm")
class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    def match(self, node: torch.fx.Node):
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
                and is_node_meta_valid(weight)
                and is_node_meta_valid(bias)
                else None
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
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
            group_biases = None  # type: ignore[assignment]
        group_biases: Optional[List[Any]]
        if all(weight is None for weight in group_weights):
            group_weights = None  # type: ignore[assignment]
        group_weights: Optional[List[Any]]
        assert all(
            eps == group_epss[0] for eps in group_epss
        ), "all epsilon values must be equal"

        with graph.inserting_before(subset[0]):
            stack_input = graph.call_function(
                torch.stack, args=(group_inputs,), kwargs={"dim": stack_dim}
            )
            if group_weights is not None:
                stack_weight = graph.call_function(
                    torch.stack, args=(group_weights,), kwargs={"dim": 0}
                )
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(
                    torch.stack, args=(group_biases,), kwargs={"dim": 0}
                )
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


class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch poinwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # for relu op, we also use the inplace to construct the key
            group_key = (
                "batch_" + self.op.__name__.lower() + "_pre_grad",
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []

        for node in subset:
            batch_nodes.append(node)
            batch_inputs.append(get_arg_value(node, 0, "input"))

        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            if self.op == torch.nn.functional.relu:
                batch_op = graph.call_function(
                    self.op,
                    args=(stack_inputs,),
                    kwargs={"inplace": subset[0].kwargs.get("inplace", False)},
                )
            else:
                batch_op = graph.call_function(
                    self.op,
                    args=(stack_inputs,),
                )
            unbind_op = graph.call_function(
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)


@register_fusion("batch_tanh")
class BatchTanhPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.tanh, **kwargs)


@register_fusion("batch_sigmoid")
class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.sigmoid, **kwargs)


@register_fusion("batch_relu")
class BatchReLuPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.nn.functional.relu, **kwargs)


def find_independent_subset_greedy(
    node_list: List[torch.fx.Node],
    graph_search_options: Dict[str, Any],
) -> Iterator[List[torch.fx.Node]]:
    """
    Return a list of subset from node_list, all nodes in each subset are independent with each other and can be fused together.
    The type of subset is list, so we can preserve node's order and benefit from split-cat elimination in later pass.
    """
    visited_node_set: Set[torch.fx.Node] = set()
    dep_set: Set[torch.fx.Node] = set()

    def find_dependent_nodes(src_node, cur_node):
        for input_node in cur_node.all_input_nodes:
            if input_node in node_list:
                dep_set.add(input_node)

            if input_node not in visited_node_set:
                visited_node_set.add(input_node)
                find_dependent_nodes(src_node, input_node)

    while len(node_list) > 0:
        subset: List[torch.fx.Node] = []
        subset_deps: Set[torch.fx.Node] = set()

        for node in node_list:
            if len(subset) >= graph_search_options["max_fuse_set_size"]:
                break

            visited_node_set.clear()
            dep_set.clear()

            find_dependent_nodes(node, node)
            if not dep_set.intersection(subset) and node not in subset_deps:
                subset.append(node)
                subset_deps.update(dep_set)

        if len(subset) >= graph_search_options["min_fuse_set_size"]:
            yield subset

        next_round_node_list = [node for node in node_list if node not in subset]
        node_list = next_round_node_list


def get_fusion_candidates(
    rule: GroupBatchFusionBase, root_node: torch.fx.Node, fused_set: Set[torch.fx.Node]
) -> DefaultDict[Any, List[torch.fx.Node]]:
    """
    Search fusion candidates for a specific rule using BFS starting from the root node.
    We only search the subgraph within graph_search_options["max_fuse_search_depth"].
    """
    q: Deque[Tuple[int, torch.fx.Node]] = collections.deque()

    candidate_dict: DefaultDict[Any, List[torch.fx.Node]] = collections.defaultdict(
        list
    )

    if root_node.target in SEARCH_EXCLUSIONS:
        return candidate_dict

    visited_set: Set[torch.fx.Node] = set()

    for next_node in root_node.all_input_nodes:
        q.append((1, next_node))
        visited_set.add(next_node)

    while len(q) > 0:
        depth, node = q.popleft()

        if node in fused_set:
            continue

        key = rule.match(node)
        # SymInt is not hashable, so we need to skip it
        if key is not None and not isinstance(key, torch.SymInt):
            candidate_nodes = candidate_dict[key]
            if node not in candidate_nodes:
                candidate_nodes.append(node)
        else:
            if depth < rule.graph_search_options["max_fuse_search_depth"]:
                for next_node in node.all_input_nodes:
                    if next_node not in visited_set:
                        visited_set.add(next_node)
                        q.append((depth + 1, next_node))

    return candidate_dict


def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase):
    stable_topological_sort(graph)
    fused_set: Set[torch.fx.Node] = set()

    for node in reversed(graph.nodes):
        candidates = get_fusion_candidates(rule, node, fused_set)

        for key, candidate_nodes in candidates.items():
            if len(candidate_nodes) < MIN_FUSE_SET_SIZE:
                continue

            for subset in find_independent_subset_greedy(
                candidate_nodes, rule.graph_search_options
            ):
                rule.fuse(graph, subset)
                fused_set.update(subset)
                if isinstance(rule, GroupFusion):
                    counters["inductor"]["group_fusion"] += 1
                elif isinstance(rule, BatchFusion):
                    counters["inductor"]["batch_fusion"] += 1
                else:
                    counters["inductor"]["unknown_group_batch_fusion"] += 1

                log.info(
                    f"{rule.__class__.__name__}: key = {key}; subset size = {len(subset)}"  # noqa: G004
                )


def generate_fusion_from_config(config_options: Dict[str, Any], pre_grad=True):
    fusions: List[GroupBatchFusionBase] = []
    for name, options in config_options.items():
        fusion_cls = PRE_GRAD_FUSIONS[name] if pre_grad else POST_GRAD_FUSIONS[name]
        _options = graph_search_options.copy()
        _options.update(options)
        fusions.append(fusion_cls(graph_search_options=_options))  # type: ignore[operator]
    return fusions


def group_batch_fusion_passes(graph: torch.fx.Graph, pre_grad=True):
    print_graph(graph, "Before group_batch fusion in pre grad pass.")
    fusions: List[GroupBatchFusionBase] = []
    # we keep all current pre grad fusions to keep
    # current implementation, will remove this later
    if pre_grad:
        fusions += generate_fusion_from_config(
            config.pre_grad_fusion_options, pre_grad=True
        )
    else:
        fbgemm_fusion_keys = [
            x
            for x in config.post_grad_fusion_options
            if config.post_grad_fusion_options[x].get("require_fbgemm", False)
        ]
        fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in fbgemm_fusion_keys
        }
        non_fbgemm_fusions = {
            fusion: config.post_grad_fusion_options[fusion]
            for fusion in config.post_grad_fusion_options.keys()
            if fusion not in fbgemm_fusion_keys
        }
        fusions += generate_fusion_from_config(non_fbgemm_fusions, pre_grad=False)
        if has_fbgemm:
            fusions += generate_fusion_from_config(fbgemm_fusions, pre_grad=False)

    for rule in fusions:
        apply_group_batch_fusion(graph, rule)
        print_graph(graph, f"Apply fusion {rule.__class__.__name__}.")
