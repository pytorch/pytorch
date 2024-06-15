# mypy: allow-untyped-defs
import collections
import logging
import operator
from collections import OrderedDict
from typing import (
    Any,
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import torch
from torch._dynamo.utils import counters, optimus_scuba_log
from torch._utils_internal import upload_graph
from torch.fx.passes.graph_transform_observer import GraphTransformObserver

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
# Whether we only fuse nodes with same parent node
FUSE_NODES_WITH_SAME_PARENT = False
# Whether we enable the add broadcast in batch linear
SHAPE_BROADCAST_BATCH_LINEAR = False
# Whether we enable the fuse nodes with same users
Fuse_NODES_WITH_SAME_USERS = False

# exclude these nodes from BFS
# excluding get item improves optimizer compilation time by 60s
SEARCH_EXCLUSIONS = {operator.getitem}


default_graph_search_options = {
    "min_fuse_set_size": MIN_FUSE_SET_SIZE,
    "max_fuse_set_size": MAX_FUSE_SET_SIZE,
    "max_fuse_search_depth": MAX_FUSE_SEARCH_DEPTH,
    "max_fuse_tensor_size_group_linear": MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR,
    "fuse_nodes_with_same_parent": FUSE_NODES_WITH_SAME_PARENT,
    "shape_broadcast_batch_linear": SHAPE_BROADCAST_BATCH_LINEAR,
    "fuse_nodes_with_same_users": Fuse_NODES_WITH_SAME_USERS,
}

graph_search_options = default_graph_search_options


def update_stack_example_value(node, metadata, dim=0, op=torch.stack):
    """
    Update the example value of the node in the graph to enable followup split cat opt.
    """
    if node is not None and hasattr(node, "meta"):
        if op == torch.stack:
            example_value = torch.stack(metadata, dim=dim)
        elif op == torch.unbind:
            example_value = torch.unbind(metadata, dim=dim)  # type: ignore[assignment]
        else:
            return
        node.meta["example_value"] = example_value


def update_pointwise_example_value(pointwise_node, input, other, op):
    """
    Update the example value of the add node in the graph to enable followup split cat opt.
    """
    if pointwise_node is not None and hasattr(pointwise_node, "meta"):
        if op == torch.add:
            example_value = torch.add(input, other)
        elif op == torch.mul:
            example_value = torch.mul(input, other)
        else:
            return
        pointwise_node.meta["example_value"] = example_value


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
    unsqueezed_inputs_meta = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(
            aten.unsqueeze, args=(input_tensor,), kwargs={"dim": 0}
        )
        unsqueezed_inputs.append(unsqueezed_input)
        unsqueezed_input.meta["val"] = aten.unsqueeze(input_tensor.meta["val"], dim=0)  # type: ignore[assignment]
        unsqueezed_inputs_meta.append(unsqueezed_input.meta["val"])
    stacked_inputs = graph.call_function(
        aten.cat, args=(unsqueezed_inputs,), kwargs={"dim": 0}
    )
    stacked_inputs.meta["val"] = aten.cat(unsqueezed_inputs_meta, dim=0)  # type: ignore[assignment]
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
        # pyre-fixme[7]: Incompatible return type
        return (
            node.kwargs.get("beta", 1.0) == 1.0 and node.kwargs.get("alpha", 1.0) == 1.0  # type: ignore[return-value]
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        input_shapes = input.meta["val"].shape
        return (
            len(input_shapes) == 2
            and isinstance(input_shapes[0], int)
            and isinstance(input_shapes[1], int)
        )

    def match(
        self, node: torch.fx.Node
    ) -> Optional[Tuple[str, int, int, int, bool, str]]:
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None
        # get the user of the node
        if self.graph_search_options.get("fuse_nodes_with_same_users", False):
            users = [user.target for user in node.users.keys()]
        else:
            users = ""  # type: ignore[assignment]
        # only handle the cases where inputs are 2D tensors
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):  # type: ignore[arg-type]
            return None
        m, k = input_m.meta["val"].shape  # type: ignore[union-attr]
        n = weight_m.meta["val"].shape[1]  # type: ignore[union-attr]
        batch_key = ("batch_linear_post_grad", m, k, n, bias_m is not None, str(users))
        return batch_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_nodes = []
        batch_inputs_meta = []
        batch_weights_meta = []
        batch_biases_meta = []

        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            elif CallFunctionVarArgs(aten.mm.default).match(node):
                input, weight = node.args
                bias = None
            batch_nodes.append(node)
            batch_inputs.append(input)  # type: ignore[possibly-undefined]
            batch_weights.append(weight)  # type: ignore[possibly-undefined]
            batch_biases.append(bias)  # type: ignore[possibly-undefined]
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_weights_meta.append(weight.meta)  # type: ignore[possibly-undefined, union-attr]
            if bias is not None:  # type: ignore[possibly-undefined]
                batch_biases_meta.append(bias.meta)  # type: ignore[possibly-undefined, union-attr]
            else:
                batch_biases_meta.append(None)

        with graph.inserting_before(subset[-1]):
            fused_inputs = decompose_stack(graph, batch_inputs)
            fused_weights = decompose_stack(graph, batch_weights)
            fused_inputs_meta_val = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            fused_weights_meta_val = torch.stack(
                [weight["val"] for weight in batch_weights_meta]
            )
            fused_bmm = graph.call_function(
                aten.bmm,
                args=(fused_inputs, fused_weights),
            )
            fused_bmm.meta["val"] = aten.bmm(
                fused_inputs_meta_val, fused_weights_meta_val
            )
        for i, original_mm in enumerate(batch_nodes):
            has_bias = False
            with graph.inserting_after(fused_bmm):
                new_mm = graph.call_function(aten.select, args=((fused_bmm, 0, i)))
                new_mm.meta["val"] = aten.select(fused_bmm.meta["val"], 0, i)
                if batch_biases[i]:
                    has_bias = True
                    # broadcast the bias to the same shape as the mm output
                    if self.graph_search_options.get(
                        "shape_broadcast_batch_linear", False
                    ):
                        broadcast_shape = torch.broadcast_shapes(
                            batch_biases_meta[i]["val"].shape, new_mm.meta["val"].shape
                        )
                        broadcast_bias = graph.call_function(
                            aten.broadcast_to.default,
                            args=(batch_biases[i],),
                            kwargs={"size": broadcast_shape},
                        )
                        broadcast_bias.meta["val"] = aten.broadcast_to(batch_biases_meta[i]["val"], broadcast_shape)  # type: ignore[assignment]
                        new_bias_add = graph.call_function(
                            aten.add.Tensor, args=((broadcast_bias, new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            broadcast_bias.meta["val"], new_mm.meta["val"]
                        )
                    else:
                        new_bias_add = graph.call_function(
                            aten.add, args=((batch_biases[i], new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            batch_biases_meta[i]["val"], new_mm.meta["val"]
                        )
            new_mm_cont = new_bias_add if has_bias else new_mm  # type: ignore[possibly-undefined]
            original_mm.replace_all_uses_with(new_mm_cont)
            new_mm_cont.meta.update(original_mm.meta)
            graph.erase_node(original_mm)
        counters["inductor"]["batch_linear_post_grad"] += 1


@register_fusion("group_linear", pre_grad=False)
class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[2].meta["val"].shape  # type: ignore[union-attr]
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
        input_shape = node.args[0].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
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

        with graph.inserting_before(subset[0]):
            fused_mm = graph.call_function(
                torch.ops.fbgemm.gmm.default,
                args=(group_inputs, group_weights, group_biases),
                kwargs={"smart_fused": True},
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)
        counters["inductor"]["group_linear"] += 1


class BatchPointwiseMathOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise math operator (e.g., add, mul) in post grad pass.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def _pointwise_node_can_be_fused(self, node: torch.fx.Node):
        # note: we only consider the case where the inputs are tensors
        # for mixed precision training, we need to make sure the inputs
        # of the aten.cat when do the stack should be the same dtype
        # otherwise, the output of the aten.cat may be not the same as
        # its inputs, and cause dtype not same error in mm or addmm
        input, other = node.args
        return (
            input.meta["val"].shape == other.meta["val"].shape  # type: ignore[union-attr]
            if hasattr(input, "meta")
            and hasattr(other, "meta")
            and "val" in input.meta  # type: ignore[union-attr]
            and "val" in other.meta  # type: ignore[union-attr]
            else False
        )

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(self.op).match(
            node
        ) and self._pointwise_node_can_be_fused(node):
            alpha = node.kwargs.get("alpha", 1.0)
            rounding_mode = node.kwargs.get("rounding_mode", None)
            input, other = node.args
            shape = list(input.meta["val"].shape)  # type: ignore[union-attr]
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # only consider the linear case so far
                # pyre-fixme[16]
                if input.target == aten.select or other.target == aten.select:  # type: ignore[union-attr]
                    parent = (
                        # pyre-fixme[16]
                        input.args[0]  # type: ignore[union-attr]
                        # pyre-fixme[16]
                        if input.target == aten.select  # type: ignore[union-attr]
                        else other.args[0]  # type: ignore[union-attr]
                    )
                else:
                    parent = ""
            else:
                parent = ""
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(shape),
                str(input.meta["val"].dtype),  # type: ignore[union-attr]
                str(other.meta["val"].dtype),  # type: ignore[union-attr]
                str(alpha),
                str(rounding_mode),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_inputs, batch_others = [], []
        alpha = subset[0].kwargs.get("alpha", 1.0)
        batch_inputs_meta, batch_others_meta = [], []

        for node in subset:
            input, other = node.args
            batch_inputs.append(input)
            batch_others.append(other)
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_others_meta.append(other.meta)  # type: ignore[possibly-undefined, union-attr]

        with graph.inserting_before(subset[0]):
            stack_inputs = decompose_stack(graph, batch_inputs)
            stack_others = decompose_stack(graph, batch_others)
            stack_inputs_meta = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            stack_others_meta = torch.stack(
                [other["val"] for other in batch_others_meta]
            )

            batch_op = graph.call_function(
                self.op,
                args=(stack_inputs, stack_others),
                kwargs={"alpha": alpha} if self.op == aten.add.Tensor else {},
            )
            batch_op.meta["val"] = self.op(stack_inputs_meta, stack_others_meta)
            for i, original_add in enumerate(subset):
                with graph.inserting_after(batch_op):
                    new_add = graph.call_function(
                        torch.ops.aten.select, args=((batch_op, 0, i))
                    )
                original_add.replace_all_uses_with(new_add)
                new_add.meta.update(original_add.meta)
                graph.erase_node(original_add)
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


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
        counters["inductor"]["batch_linear_lhs"] += 1


def is_node_meta_valid(node: Optional[torch.fx.Node]):
    if node is None:
        return True
    if "example_value" not in node.meta and "val" not in node.meta:
        return False
    return True


# Poor person's check for if a node in the graph mutates its input.
# (the graph is torch IR, so we will see torch fns and python operators)
def _is_mutable_node(tgt):
    if str(tgt).endswith("_"):
        # e.g. torch.mul_, torch.Tensor.mul_
        return True
    if (
        hasattr(tgt, "__module__")
        and tgt.__module__ == "_operator"
        and tgt.__name__.startswith("i")
    ):
        # e.g. operator.iand, operator.imul
        return True
    return False


def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    return (
        is_node_meta_valid(node)
        and is_node_meta_valid(input)
        and is_node_meta_valid(weight)
        and len(input.meta["example_value"].shape) == 2
        and len(weight.meta["example_value"].shape) == 2
        # the mm -> bmm transform adds an unbind() op,
        # which is not safe for autograd when the output of the mm is mutated.
        # don't pattern match if any users of the mm mutate the input.
        and not any(_is_mutable_node(user.target) for user in node.users)
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
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users.keys()]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                "batch_linear",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
                str(users),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_inputs_metadata = []
        batch_weights_metadata = []
        batch_biases_metadata = []
        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])
            weight = get_arg_value(node, 1, "weight")
            batch_weights.append(weight)
            batch_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 2, "bias")
            batch_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                batch_biases_metadata.append(bias.meta["example_value"])

        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            stack_weights = graph.call_function(
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_weights, batch_weights_metadata)
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
                update_stack_example_value(stack_biases, batch_biases_metadata)
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
        counters["inductor"]["batch_linear"] += 1


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
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users.keys()]
            else:
                users = ""  # type: ignore[assignment]
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
                    str(users),
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
        group_inputs_metadata = []
        group_biases_metadata = []
        group_weights_metadata = []
        for node in subset:
            group_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            group_inputs.append(input)
            group_inputs_metadata.append(input.meta["example_value"])
            group_shapes.append(get_arg_value(node, 1, "normalized_shape"))
            weight = get_arg_value(node, 2, "weight")
            group_weights.append(weight)
            if weight is not None and hasattr(weight, "meta"):
                group_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 3, "bias")
            group_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                group_biases_metadata.append(bias.meta["example_value"])
            eps = get_arg_value(node, 4, "eps")
            if eps is None:
                eps = 1e-5
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]
        if all(weight is None for weight in group_weights):
            group_weights = None  # type: ignore[assignment]
        assert all(
            eps == group_epss[0] for eps in group_epss
        ), "all epsilon values must be equal"

        with graph.inserting_before(subset[0]):
            stack_input = graph.call_function(
                torch.stack, args=(group_inputs,), kwargs={"dim": stack_dim}
            )
            update_stack_example_value(stack_input, group_inputs_metadata, stack_dim)
            if group_weights is not None:
                stack_weight = graph.call_function(
                    torch.stack, args=(group_weights,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_weight, group_weights_metadata)
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(
                    torch.stack, args=(group_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_bias, group_biases_metadata)
            else:
                stack_bias = None

            batch_layer_norm = graph.call_function(
                torch.nn.functional.layer_norm,
                args=(stack_input, group_shapes[-1]),
                kwargs={"eps": group_epss[-1]},
            )
            batch_layer_norm.meta["example_value"] = stack_input.meta["example_value"]

            if group_weights is not None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )
            elif group_weights is not None and group_biases is None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
            elif group_weights is None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )

            batch_layer_norm_unbind = graph.call_function(
                torch.unbind,
                args=(batch_layer_norm,),
                kwargs={"dim": stack_dim},
            )
            update_stack_example_value(
                batch_layer_norm_unbind,
                batch_layer_norm.meta["example_value"],
                op=torch.unbind,
                dim=stack_dim,
            )

        for i, node in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):
                new_node = graph.call_function(
                    operator.getitem, args=(batch_layer_norm_unbind, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)
        counters["inductor"]["batch_layernorm"] += 1


class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # pyre-fixme[16]
                parent = node.args[0]
                parent = parent.target if parent is not None else ""  # type: ignore[union-attr]
            else:
                parent = ""
            # for relu op, we also use the inplace to construct the key
            group_key = (
                "batch_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
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
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in post grad pass.
    The introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # for relu op, we also use the inplace to construct the key
            # we batch the ops with same parent to enable followup split cat
            parent = node.args[0]
            parent = parent.target if self.graph_search_options.get("fuse_nodes_with_same_parent", False) else ""  # type: ignore[union-attr]
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["val"].shape),
                str(node.kwargs.get("inplace", False)),
                # pyre-fixme[16]
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["val"])

        with graph.inserting_before(subset[0]):
            stack_inputs = decompose_stack(graph, batch_inputs)
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(
                self.op,
                args=(stack_inputs,),
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(batch_op):
                    getitem = graph.call_function(aten.select, args=(batch_op, 0, i))
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


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


@register_fusion("batch_aten_tanh", pre_grad=False)
class BatchTanhPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.tanh.default, **kwargs)


@register_fusion("batch_aten_sigmoid", pre_grad=False)
class BatchSigmoidPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.sigmoid.default, **kwargs)


@register_fusion("batch_aten_relu", pre_grad=False)
class BatchReLuPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.relu.default, **kwargs)


@register_fusion("batch_aten_add", pre_grad=False)
class BatchAddPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.add.Tensor, **kwargs)


@register_fusion("batch_aten_sub", pre_grad=False)
class BatchSubPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.sub.Tensor, **kwargs)


@register_fusion("batch_aten_div", pre_grad=False)
class BatchDivPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.div.Tensor, **kwargs)


@register_fusion("batch_aten_mul", pre_grad=False)
class BatchMulPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs):
        super().__init__(aten.mul.Tensor, **kwargs)


class _OrderedSet:
    def __init__(self, param=None):
        if param:
            self.rep = OrderedDict(dict.fromkeys(param))
        else:
            self.rep = OrderedDict()

    def __contains__(self, o):
        return o in self.rep

    def __len__(self):
        return self.rep.__len__()

    def append(self, o):
        self.rep[o] = None

    def __iter__(self):
        return self.rep.keys().__iter__()


def find_independent_subset_greedy(
    node_list: Iterable[torch.fx.Node],
    graph_search_options: Dict[str, Any],
) -> Iterator[Iterable[torch.fx.Node]]:
    """
    Yields a list of subsets of `node_list` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_list` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_options:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """

    # Compute all the children of `node` which are members of
    # `interesting_nodes`.
    def find_dependent_nodes(node, interesting_nodes):
        visited_node_set: Set[torch.fx.Node] = {node}
        dep_set: Set[torch.fx.Node] = set()

        work = [node]
        while work:
            node = work.pop()
            for input_node in node.all_input_nodes:
                if input_node in interesting_nodes:
                    dep_set.add(input_node)

                if input_node not in visited_node_set:
                    visited_node_set.add(input_node)
                    work.append(input_node)

        return dep_set

    min_fuse_set_size = graph_search_options["min_fuse_set_size"]
    max_fuse_set_size = graph_search_options["max_fuse_set_size"]

    # node_list needs to be a set because we only track the nodes that are left
    # in it (and we want to do the `in` on a set, not a list). But we want to
    # keep the correct order.
    node_list = _OrderedSet(node_list)

    cache: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}
    while node_list:
        subset: List[torch.fx.Node] = []
        subset_deps: Set[torch.fx.Node] = set()

        next_round_node_list = _OrderedSet()
        for node in node_list:
            if len(subset) >= max_fuse_set_size or node in subset_deps:
                next_round_node_list.append(node)
                continue

            dep_set = cache.pop(node, None)
            if dep_set is None:
                dep_set = find_dependent_nodes(node, node_list)

            if not dep_set.intersection(subset):
                subset.append(node)
                subset_deps.update(dep_set)
            else:
                next_round_node_list.append(node)
                cache[node] = dep_set

        if len(subset) >= min_fuse_set_size:
            # Careful here - the caller uses the subsets to fuse nodes together
            # so we need to clear any cache entry that contains one of the
            # returned nodes because the dependency list could be different
            # (larger) after the merge.
            cache = {k: v for k, v in cache.items() if v.isdisjoint(subset)}
            yield subset

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
        if key is not None:
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
    stable_topological_sort(graph)  # type: ignore[arg-type]
    fused_set: Set[torch.fx.Node] = set()
    log_to_scuba = False

    for node in reversed(graph.nodes):
        candidates = get_fusion_candidates(rule, node, fused_set)

        for key, candidate_nodes in candidates.items():
            if len(candidate_nodes) < rule.graph_search_options["min_fuse_set_size"]:
                continue

            for subset in find_independent_subset_greedy(
                candidate_nodes, rule.graph_search_options
            ):
                rule.fuse(graph, subset)
                fused_set.update(subset)
                log.debug(
                    f"{rule.__class__.__name__}: key = {key}; subset size = {len(list(subset))}"  # noqa: G004
                )
                log_to_scuba = True
    if log_to_scuba:
        optimus_scuba_log[rule.__class__.__name__] = upload_graph(graph)


def generate_fusion_from_config(config_options: Dict[str, Any], pre_grad=True):
    fusions: List[GroupBatchFusionBase] = []
    for name, options in config_options.items():
        # we skip all patterns from pattern_matcher passes (e.g., split_cat)
        if name not in PRE_GRAD_FUSIONS and name not in POST_GRAD_FUSIONS:
            continue
        fusion_cls = PRE_GRAD_FUSIONS[name] if pre_grad else POST_GRAD_FUSIONS[name]
        _options = graph_search_options.copy()
        _options.update(options)
        fusions.append(fusion_cls(graph_search_options=_options))  # type: ignore[operator]
    return fusions


def group_batch_fusion_passes(graph: torch.fx.Graph, pre_grad=True):
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

    for i, rule in enumerate(fusions):
        with GraphTransformObserver(
            graph.owning_module,
            f"group_batch_fusion_{i}",
            config.trace.log_url_for_graph_xform,
        ):
            apply_group_batch_fusion(graph, rule)  # type: ignore[arg-type]
