import torch
from torch.fx import GraphModule
from torch.nn.utils.fusion import fuse_conv_bn_weights
# TODO[jerryzh168]: move this to a more general util function
from torch.ao.quantization.fx.prepare import (
    _is_activation_post_process_node,
)
import operator


def _get_tensor_constant_from_node(node, m):
    if node is None:
        return None
    assert node.op == "get_attr"
    return getattr(m, node.target)

# fuse conv bn weights, inplace modification of the graph_module and graph
def _fuse_conv_bn_(m: GraphModule) -> None:
    for n in m.graph.nodes:
        if n.op != "call_function" or n.target != torch.ops.aten.native_batch_norm.default:
            continue
        bn_op = n
        n = bn_op.args[0]
        if n.op != "call_function" or n.target != torch.ops.aten.convolution.default:
            continue
        conv_op = n

        # conv weight
        conv_w = _get_tensor_constant_from_node(conv_op.args[1], m)
        # conv bias
        conv_b = _get_tensor_constant_from_node(conv_op.args[2], m)
        transpose = conv_op.args[6]

        # bn weight
        bn_w = _get_tensor_constant_from_node(bn_op.args[1], m)
        # bn bias
        bn_b = _get_tensor_constant_from_node(bn_op.args[2], m)
        # bn running mean
        bn_rm = _get_tensor_constant_from_node(bn_op.args[3], m)
        # bn running variance
        bn_rv = _get_tensor_constant_from_node(bn_op.args[4], m)
        bn_eps = bn_op.args[7]

        fused_weight, fused_bias = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False)

        # update the weight and bias for conv
        conv_args = list(conv_op.args)
        # calling data since the fused_weight and fused_bias are nn.Parameter
        weight_attr_name = conv_args[1].target
        setattr(m, weight_attr_name, fused_weight)
        if conv_args[2] is not None:
            bias_attr_name = conv_args[2].target
        else:
            bias_attr_name = weight_attr_name + "_bias"
            with m.graph.inserting_before(conv_op):
                get_bias_node = m.graph.get_attr(bias_attr_name)
            conv_args[2] = get_bias_node
        setattr(m, bias_attr_name, fused_bias)
        conv_op.args = tuple(conv_args)

        # native_batch_norm has 3 outputs, we expect getitem calls on the output
        # and we want to replace the uses of getitem 0 with the output of conv
        #
        # Before:
        # conv -> bn - (first output) -> users1
        #          \ - (second output) -> users2
        #          \ - (third output) -> users3
        # After:
        # conv -> (first output) -> users1
        #       bn -
        #          \ - (second output) -> users2
        #          \ - (third output) -> users3
        # if users2 and users3 are empty then bn will be removed through dead code elimination

        for user in bn_op.users:
            if user.op != "call_function" or user.target != operator.getitem or user.args[1] != 0:
                continue
            user.replace_all_uses_with(conv_op)
    m.graph.eliminate_dead_code()
    m.recompile()

# TODO: remove hack when we have better support for pattern matching
# move around the observer for addmm
def _rearrange_weight_observer_for_decomposed_linear(
    model: GraphModule,
) -> None:
    """
    Linear is decomposed to `t - addmm` (w/ bias) or `t - mm` (w/o bias)
    before:
         weight - t - observer \
           input - observer - addmm/mm
    after:
         weight - observer - t \
           input - observer - addmm/mm
    """
    aten = torch.ops.aten
    op_to_weight_obs_index = {
        aten.addmm.default : 2,
        aten.mm.default : 1,
    }
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.target not in (aten.addmm.default, aten.mm.default):
            continue
        root_node = node
        maybe_weight_obs = root_node.args[op_to_weight_obs_index[root_node.target]]
        if not _is_activation_post_process_node(maybe_weight_obs, named_modules):
            continue
        transpose_node = maybe_weight_obs.args[0]
        if transpose_node.target != torch.ops.aten.t.default:
            continue
        # swap the order of transpose and observation

        maybe_weight_obs.replace_input_with(transpose_node, transpose_node.args[0])
        # remove the transpose node
        with model.graph.inserting_after(maybe_weight_obs):
            args = list(transpose_node.args)
            args[0] = maybe_weight_obs
            new_transpose_node = model.graph.create_node(
                "call_function",
                torch.ops.aten.t.default,
                tuple(args),
                transpose_node.kwargs
            )
        root_node.replace_input_with(maybe_weight_obs, new_transpose_node)

    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()
