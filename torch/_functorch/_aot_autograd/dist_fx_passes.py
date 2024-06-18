import math
import torch
import operator
from collections import defaultdict
from torch._prims_common import make_contiguous_strides_for


def _flatten_arg_list(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat_args.extend(_flatten_arg_list(arg))
        else:
            flat_args.append(arg)
    return flat_args


def _collect_primal_inputs_used_by_set_op(node_list):
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, node_list)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    primal_inputs_used = set()
    primal_set_info_dict = {}
    for i, n in enumerate(node_list):
        if n.target == torch.ops.aten.set_.source_Tensor and n.args[0] in primal_inputs_tensor_only:
            primal_input = n.args[0]
            set_node = n
            primal_set_info_dict[primal_input] = set_node
            primal_inputs_used.add(primal_input)
    return primal_inputs_used, primal_set_info_dict


def return_primal_instead_of_view(mod):
    """
    view_5: "f32[524288]" = torch.ops.aten.view.default(getitem_4, [524288]);  getitem_4 = None
    as_strided_19: "f32[1024, 512]" = torch.ops.aten.as_strided.default(view_5, [1024, 512], [512, 1], 0);  view_5 = None
    set_: "f32[1024, 512]" = torch.ops.aten.set_.source_Tensor(primals_5, as_strided_19);  primals_5 = as_strided_19 = None
    return [..., view_5, ...]

    -> 

    view_5: "f32[524288]" = torch.ops.aten.view.default(getitem_4, [524288]);  getitem_4 = None
    as_strided_19: "f32[1024, 512]" = torch.ops.aten.as_strided.default(view_5, [1024, 512], [512, 1], 0);  view_5 = None
    set_: "f32[1024, 512]" = torch.ops.aten.set_.source_Tensor(primals_5, as_strided_19);  primals_5 = as_strided_19 = None
    return [..., primals_5, ...]
    """
    node_list = list(mod.graph.nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    return_op = None
    for node in node_list:
        if node.target == "output":
            return_op = node
            break
    _, primal_set_info_dict = _collect_primal_inputs_used_by_set_op(node_list)
    return_op_new_args = return_op.args[0][:]
    for primal_node in primal_set_info_dict:
        set_node = primal_set_info_dict[primal_node]
        as_strided_node = set_node.args[1]
        view_node = as_strided_node.args[0]
        as_strided_shape = as_strided_node.args[1]
        as_strided_stride = as_strided_node.args[2]
        # Make sure the as_strided node is truly a no-op
        if not (
            math.prod(view_node.meta["tensor_meta"].shape) == math.prod(as_strided_node.meta["tensor_meta"].shape)
            and list(make_contiguous_strides_for(as_strided_shape)) == as_strided_stride
        ):
            continue
        # Return the primal node instead of the view node from the graph
        try:
            idx = return_op_new_args.index(view_node)
            return_op_new_args[idx] = primal_node
        except ValueError:
            continue
        return_op.args = (return_op_new_args,)
    mod.graph.lint()
    mod.recompile()


def _is_fsdp_allgather_copyout(node):
    """
    auto_functionalized = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_4, all_gather_input_split_sizes = [262144, 256, 262144], dim = 1, out = [view_1, view_2, view_3]);  view_4 = view_1 = view_2 = view_3 = None
    getitem_3 = auto_functionalized[1];  auto_functionalized = None
    getitem_5: "f32[2, 256]" = getitem_3[1]
    view_7: "f32[512]" = torch.ops.aten.view.default(getitem_5, [512])
    as_strided_20: "f32[512]" = torch.ops.aten.as_strided.default(view_7, [512], [1], 0);  view_7 = None
    set__1: "f32[512]" = torch.ops.aten.set_.source_Tensor(primals_6, as_strided_20);  primals_6 = as_strided_20 = None
    return [..., getitem_5, ...]

    In this case, `view_7` is a FSDP AllGather copy-out node.
    """
    if (
        node.target == torch.ops.aten.view.default
        and node.args[0].target == operator.getitem
        and node.args[0].args[0].target == operator.getitem
        and node.args[0].args[0].args[0].target == torch._higher_order_ops.auto_functionalize.auto_functionalized
        and node.args[0].args[0].args[0].args[0] == torch.ops.fsdp.split_with_sizes_copy.default
    ):
        return True
    return False


def should_ban_recomputation(node):
    if _is_fsdp_allgather_copyout(node):
        # This combined with `return_primal_instead_of_view` pass prevents alias of primals
        # from being saved as output of FWD graph.
        # (Normally we would expect AOTAutograd to be able to dedup aliases,
        # but AOTAutograd's alias dedup logic cannot support subclass + alias + mutation very well yet.)
        return True
    return False
