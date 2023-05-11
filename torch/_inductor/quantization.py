import torch
from .fx_passes.post_grad import register_lowering_pattern
from .ir import QConv
from .pattern_matcher import Arg, CallFunction, Match


def _is_cpu(example_inputs):
    return all(
        example_input.device == torch.device("cpu")
        for example_input in example_inputs
        if isinstance(example_input, torch.Tensor)
    )


def _is_quantized_graph_module(gm: torch.fx.GraphModule):
    found_quantize = False
    quantize_ops = (
        torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    )
    for node in gm.graph.nodes:
        if node.target in quantize_ops:
            found_quantize = True
            break
    return found_quantize


def _quantize_and_replace_weight(
    gm: torch.fx.GraphModule, dq_per_channel_node: torch.fx.Node
):
    # pattern: w - q - dq - weighted op
    # after: qw - dq - weighted op
    q_per_channel_node = dq_per_channel_node.args[0]
    weight_node = q_per_channel_node.args[0]
    w_attr_name = weight_node.target
    weight = getattr(gm, w_attr_name)

    assert isinstance(weight, torch.Tensor), "Cannot find weight for quantization"
    if weight.is_quantized:
        return
    quantize_args = (
        getattr(gm, n.target) if isinstance(n, torch.fx.Node) else n
        for n in q_per_channel_node.args
    )
    q_arg_list = list(quantize_args)
    q_arg_tuple = tuple(q_arg_list)
    weight_int8 = torch.ops.quantized_decomposed.quantize_per_channel(*q_arg_tuple)

    qw_attr_name = w_attr_name + "_quant"
    setattr(gm, qw_attr_name, weight_int8)
    weight_node.target = qw_attr_name
    gm.graph.owning_module._buffers[qw_attr_name] = weight_int8
    delattr(gm, w_attr_name)
    q_per_channel_node.replace_all_uses_with(weight_node)
    gm.graph.erase_node(q_per_channel_node)


def _pre_quantize_weights(gm: torch.fx.GraphModule):
    # pattern: w - q - dq - weighted op
    # after: qw - dq - weighted op
    aten = torch.ops.aten
    decomposed = torch.ops.quantized_decomposed
    for node in gm.graph.nodes:
        dq_per_channel_node = None
        if node.target == aten.convolution.default:
            # conv args = (x, w, ...)
            dq_per_channel_node = node.args[1]
        if dq_per_channel_node is not None:
            assert (
                dq_per_channel_node.target == decomposed.dequantize_per_channel
            ), "Cannot find the dequantize op for weight"
            _quantize_and_replace_weight(gm, dq_per_channel_node)
    gm.graph.lint()
    gm.recompile()


def quantization_pre_grad_pass(gm: torch.fx.GraphModule, example_inputs):
    # skip if gm is not a quantized graph module
    if not (_is_quantized_graph_module(gm) and _is_cpu(example_inputs)):
        return gm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # Fuse `quant_per_channel - weight` and replace the original fp32 weight with quantized one
    _pre_quantize_weights(gm)

    return gm


aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
dequantize_per_channel = quantized_decomposed.dequantize_per_channel.default

"""
dequantize activation:
    x = x.to(fp32)
    x = x - zero_point
    x = x * scale
"""
dequantize_activation_pattern = CallFunction(
    aten.mul.Tensor,
    CallFunction(
        aten.sub.Tensor,
        CallFunction(
            prims.convert_element_type.default, Arg(), Arg()  # args = x, torch.float32
        ),
        Arg(),  # zero point
    ),
    Arg(),  # scales
)

dequantize_weight_pattern = CallFunction(
    dequantize_per_channel,
    Arg(),  # weight
    Arg(),  # scales
    Arg(),  # zero point
    Arg(),  # axis
    Arg(),  # lower limit
    Arg(),  # upper limit
    Arg(),  # dtype=torch.int8
)

aten_conv_pattern = CallFunction(
    aten.convolution.default,
    dequantize_activation_pattern,
    dequantize_weight_pattern,
    Arg(),  # bias
    Arg(),  # stride
    Arg(),  # padding
    Arg(),  # dilation
    Arg(),  # transposed
    Arg(),  # output_padding
    Arg(),  # groups
)

"""
quantize output:
    scale = 1 / scale
    scale = 1.0 * scale
    output = round(output * scale)
    output = output + zero_point
    output = clamp_min(output, 0)
    output = clamp_max(output, 127)
    output = output.to(uint8)
"""
quantize_conv_output_pattern = CallFunction(
    prims.convert_element_type.default,
    CallFunction(
        aten.clamp_max.default,
        CallFunction(
            aten.clamp_min.default,
            CallFunction(
                aten.add.Tensor,
                CallFunction(
                    aten.round.default,
                    CallFunction(
                        aten.mul.Tensor,
                        aten_conv_pattern,  # output of conv
                        CallFunction(
                            aten.mul.Tensor,
                            CallFunction(
                                aten.reciprocal.default, Arg()  # arg = scales
                            ),
                            Arg(),  # 1.0
                        ),
                    ),
                ),
                Arg(),  # zero point
            ),
            Arg(),  # 0
        ),
        Arg(),  # 127
    ),
    Arg(),  # dtype=torch.uint8
)


@register_lowering_pattern(quantize_conv_output_pattern)
def qconv(match: Match, *args, **kwargs):
    """
    There are 24 args. They are
    [0] input
    [1] input dequant dtype (e.g., float32)
    [2] input zero point
    [3] input scale
    [4] weight
    [5] weight scales
    [6] weight zero points
    [7] weight channel axis (e.g., 0)
    [8] weight quant min (e.g., -128)
    [9] weight quant max (e.g., 127)
    [10] weight quant dtype (e.g., int8)
    [11] conv bias
    [12] conv stride
    [13] conv padding
    [14] conv dilation
    [15] transposed (= False)
    [16] conv output padding (unused for conv)
    [17] groups
    [18] output scale
    [19] output scale coefficient = 1.0
    [20] output zero point
    [21] output quant min (e.g., 0)
    [22] output quant max (e.g., 127 with reduce_range=True)
    [23] output quant dtype (e.g., uint8)
    """
    x, x_scale, x_zp = args[0], args[3], args[2]
    w, w_scale, w_zp, w_axis = args[4], args[5], args[6], args[7]
    b = args[11]
    stride, padding, dilation = args[12], args[13], args[14]
    groups, o_scale, o_zero_point, o_dtype = args[17], args[18], args[20], args[23]
    weight_shape = w.get_size()
    dim = len(weight_shape) - 2
    return QConv.create(
        dim,
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        w_axis,
        b,
        stride,
        padding,
        dilation,
        groups,
        o_scale,
        o_zero_point,
        o_dtype,
    )
