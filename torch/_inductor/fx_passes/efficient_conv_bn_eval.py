# mypy: allow-untyped-defs
import torch
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.func import functional_call

from ..pattern_matcher import (
    CallFunctionVarArgs,
    CallModuleVarArgs,
    Match,
    register_graph_pattern,
)
from .pre_grad import efficient_conv_bn_eval_pass


def efficient_conv_bn_eval(
    bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor
):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.
        conv (nn.modules.conv._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """

    assert bn.running_var is not None
    assert bn.running_mean is not None

    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    target_shape = [-1] + [1] * (conv.weight.ndim - 1)
    if isinstance(conv, nn.modules.conv._ConvTransposeNd):
        # for transposed conv, the C_out dimension should at index 1.
        target_shape[:2] = [target_shape[1], target_shape[0]]
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (
        bias_on_the_fly - bn.running_mean
    )

    input = x
    params = {"weight": weight_on_the_fly, "bias": bias_on_the_fly}
    output = functional_call(conv, params, input)
    return output


def efficient_conv_bn_eval_decomposed(
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
    bn_eps,
    conv: torch._ops.OpOverload,
    conv_weight,
    conv_bias,
    x,
    conv_remainging_args,
):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
    """
    assert bn_running_var is not None

    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv_weight
    if conv_bias is not None:
        bias_on_the_fly = conv_bias
    else:
        bias_on_the_fly = torch.zeros_like(bn_running_var)

    if bn_weight is not None:
        bn_weight = bn_weight
    else:
        bn_weight = torch.ones_like(bn_running_var)

    if bn_bias is not None:
        bn_bias = bn_bias
    else:
        bn_bias = torch.zeros_like(bn_running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    target_shape = [-1] + [1] * (conv_weight.ndim - 1)
    if "conv_transpose" in conv.__str__():
        # for transposed conv, the C_out dimension should at index 1.
        target_shape[:2] = [target_shape[1], target_shape[0]]
    weight_coeff = torch.rsqrt(bn_running_var + bn_eps).reshape(target_shape)
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (
        bias_on_the_fly - bn_running_mean
    )

    input = x
    return conv(*((input, weight_on_the_fly, bias_on_the_fly) + conv_remainging_args))


@register_graph_pattern(
    CallFunctionVarArgs(
        [
            torch.nn.functional.batch_norm,
        ]
    ),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform_inlined(match: Match, *args, **kwargs):
    bn_node = match.nodes[0]
    graph = match.graph
    assert len(bn_node.args) == 8

    # We can only use efficient conv-bn for eval mode with track_running_stats
    # bn_node.args is `training`
    if bn_node.args[-3]:
        return

    # Check if the input is Conv
    input_node = bn_node.args[0]

    if input_node.op != "call_function":  # type: ignore[union-attr]
        return

    input_fn = input_node.target  # type: ignore[arg-type, union-attr]
    supported_convs = [
        torch._C._nn.linear,
        torch.conv1d,
        torch.conv2d,
        torch.conv3d,
        torch.conv_transpose1d,
        torch.conv_transpose2d,
        torch.conv_transpose3d,
    ]

    if not any(input_fn is cls for cls in supported_convs):
        return

    conv_node = input_node
    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(bn_node):
        # prepare args for the fused function
        bn_running_mean = bn_node.args[1]
        bn_running_var = bn_node.args[2]
        bn_weight = bn_node.args[3]
        bn_bias = bn_node.args[4]
        bn_eps = bn_node.args[7]
        assert len(conv_node.args) >= 2  # type: ignore[union-attr]
        conv_input = conv_node.args[0]  # type: ignore[union-attr]
        conv_weight = conv_node.args[1]  # type: ignore[union-attr]
        conv_bias = conv_node.args[2] if len(conv_node.args) >= 3 else None  # type: ignore[union-attr]
        conv_remainging_args = conv_node.args[3:]  # type: ignore[union-attr]
        args = (
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            conv_node.target,  # type: ignore[union-attr]
            conv_weight,
            conv_bias,
            conv_input,
            conv_remainging_args,
        )

        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval_decomposed,
            args=args,  # type: ignore[arg-type]
            name="efficient_conv_bn_eval",
        )

    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)  # type: ignore[arg-type]

    return


@register_graph_pattern(
    CallFunctionVarArgs(
        [
            torch.ops.aten.batch_norm.default,
        ]
    ),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform_decomposed(match: Match, *args, **kwargs):
    bn_node = match.nodes[0]
    graph = match.graph
    assert len(bn_node.args) == 9

    # We can only use efficient conv-bn for eval mode with track_running_stats
    # bn_node.args is `training`
    if bn_node.args[-4]:
        return

    # Check if the input is Conv
    input_node = bn_node.args[0]

    if input_node.op != "call_function":  # type: ignore[union-attr]
        return

    input_fn = input_node.target  # type: ignore[arg-type, union-attr]
    supported_convs = [
        torch.ops.aten.linear.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.conv_transpose3d.input,
    ]

    if not any(input_fn is cls for cls in supported_convs):
        return

    conv_node = input_node
    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(bn_node):
        # prepare args for the fused function
        bn_weight = bn_node.args[1]
        bn_bias = bn_node.args[2]
        bn_running_mean = bn_node.args[3]
        bn_running_var = bn_node.args[4]
        bn_eps = bn_node.args[7]
        assert len(conv_node.args) >= 2  # type: ignore[union-attr]
        conv_input = conv_node.args[0]  # type: ignore[union-attr]
        conv_weight = conv_node.args[1]  # type: ignore[union-attr]
        conv_bias = conv_node.args[2] if len(conv_node.args) >= 3 else None  # type: ignore[union-attr]
        conv_remainging_args = conv_node.args[3:]  # type: ignore[union-attr]
        args = (
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            conv_node.target,  # type: ignore[union-attr]
            conv_weight,
            conv_bias,
            conv_input,
            conv_remainging_args,
        )

        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval_decomposed,
            args=args,  # type: ignore[arg-type]
            name="efficient_conv_bn_eval",
        )

    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)  # type: ignore[arg-type]

    return


@register_graph_pattern(
    CallModuleVarArgs(
        [
            nn.modules.batchnorm._BatchNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        ],
    ),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: not inductor_config.freezing
    and inductor_config.efficient_conv_bn_eval_fx_passes,
)
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs):
    # We matched a BN node
    bn_node = match.nodes[0]
    graph = match.graph
    gm = graph.owning_module
    bn_mod = getattr(gm, bn_node.target)  # type: ignore[arg-type]

    # We can only use efficient conv-bn for eval mode with track_running_stats
    if not bn_mod.track_running_stats or bn_mod.training:
        return

    # Check if the input is Conv
    if bn_node.args:
        input_node = bn_node.args[0]
    else:
        input_node = bn_node.kwargs["input"]
    if input_node.op != "call_module":  # type: ignore[union-attr]
        return
    if not hasattr(gm, input_node.target):  # type: ignore[arg-type, union-attr]
        return
    input_mod = getattr(gm, input_node.target)  # type: ignore[arg-type, union-attr]
    supported_convs = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]
    if not any(isinstance(input_mod, cls) for cls in supported_convs):
        return
    conv_node = input_node
    # Output of conv is used by other nodes, cannot optimize
    if len(conv_node.users) > 1:  # type: ignore[union-attr]
        return

    # Find a pair of conv and bn computation nodes to optimize.
    counters["inductor"]["efficient_conv_bn_eval"] += 1

    with graph.inserting_before(conv_node):  # type: ignore[arg-type]
        # create `get_attr` node to access modules
        # note that we directly call `create_node` to fill the `name`
        # argument. `graph.get_attr` and
        # `graph.call_function` does not allow the `name` argument.
        conv_get_node = graph.create_node(
            op="get_attr",
            target=conv_node.target,  # type: ignore[union-attr]
            name="get_conv",
        )
        bn_get_node = graph.create_node(
            op="get_attr", target=bn_node.target, name="get_bn"
        )
        if conv_node.args:  # type: ignore[union-attr]
            conv_input = conv_node.args[0]  # type: ignore[union-attr]
        else:
            conv_input = conv_node.kwargs["input"]  # type: ignore[union-attr]
        # prepare args for the fused function
        args = (bn_get_node, conv_get_node, conv_input)
        # create a new node
        new_node = graph.create_node(
            op="call_function",
            target=efficient_conv_bn_eval,
            args=args,
            name="efficient_conv_bn_eval",
        )
    # this node replaces the original conv + bn, and therefore
    # should replace the uses of bn_node
    bn_node.replace_all_uses_with(new_node)
    # take care of the deletion order:
    # delete bn_node first, and then conv_node
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)  # type: ignore[arg-type]
