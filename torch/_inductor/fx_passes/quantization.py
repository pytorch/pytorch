import functools

import torch
from ..ir import QConv
from ..pattern_matcher import Arg, CallFunction, KeywordArg, Match
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern

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
            prims.convert_element_type.default,
            KeywordArg("x"),
            KeywordArg("x_dq_dtype"),
        ),
        KeywordArg("x_zp"),
    ),
    KeywordArg("x_scale"),
)

dequantize_weight_pattern = CallFunction(
    dequantize_per_channel,
    KeywordArg("w"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),  # axis for quantization
    KeywordArg("w_qmin"),  # quant clamp min
    KeywordArg("w_qmax"),  # quant clamp max
    KeywordArg("qw_dtype"),  # dtype=torch.int8
)

aten_conv_pattern = CallFunction(
    aten.convolution.default,
    dequantize_activation_pattern,
    dequantize_weight_pattern,
    KeywordArg("b"),  # bias
    KeywordArg("stride"),
    KeywordArg("padding"),
    KeywordArg("dilation"),
    KeywordArg("transposed"),
    KeywordArg("o_padding"),
    KeywordArg("groups"),
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
                                aten.reciprocal.default, KeywordArg("o_scale")
                            ),
                            Arg(),  # 1.0
                        ),
                    ),
                ),
                KeywordArg("o_zp"),
            ),
            KeywordArg("o_qmin"),  # 0
        ),
        KeywordArg("o_qmax"),  # 127
    ),
    KeywordArg("o_dtype"),  # dtype=torch.uint8
)


dequant_per_channel_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,  # dequant_per_channel node
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)


dequant_per_channel_clone_to_channel_last_pattern = CallFunction(
    aten.clone.default,
    dequant_per_channel_pattern,
    memory_format=KeywordArg("memory_format"),
)


def _register_quantized_conv_lowering(pattern):
    @register_lowering_pattern(pattern)
    def qconv(match: Match, *args, **kwargs):
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        w, w_scale, w_zp, w_axis = (
            kwargs["w"],
            kwargs["w_scale"],
            kwargs["w_zp"],
            kwargs["w_axis"],
        )
        b, stride, padding, dilation = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
        )
        groups, o_scale, o_zero_point, o_dtype = (
            kwargs["groups"],
            kwargs["o_scale"],
            kwargs["o_zp"],
            kwargs["o_dtype"],
        )
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

    return qconv


def register_quantization_lowerings():
    _register_quantized_conv_lowering(quantize_conv_output_pattern)


def _is_valid_dequant_conv2d_pattern(match):
    # Here we do some further check to ensure:
    # 1. It's a conv2d node with dim of 4, since we only support lowering of conv2d now.
    # 2. The dequant pattern has only 1 user of conv2d node.
    # If these conditions don't meet, we will not
    # insert weight prepack node into the matched pattern.
    conv_node = match.output_node()
    assert conv_node.target is aten.convolution.default
    input_meta_value = conv_node.args[0].meta.get("val")
    weight_meta_value = conv_node.args[1].meta.get("val")
    for meta_value in [input_meta_value, weight_meta_value]:
        if (
            meta_value is None
            or meta_value.device.type != "cpu"
            or meta_value.dim() != 4
        ):
            # Only support conv2d now
            return False

    mul_node = conv_node.args[0]
    sub_node = mul_node.args[0]
    to_fp32_node = sub_node.args[0]

    assert to_fp32_node.target is prims.convert_element_type.default
    assert sub_node.target is aten.sub.Tensor
    assert mul_node.target is aten.mul.Tensor
    if (
        len(list(to_fp32_node.users)) != 1
        or len(list(sub_node.users)) != 1
        or len(list(mul_node.users)) != 1
    ):
        # Ensure the dequant pattern only has 1 user
        # since we will delete the dequant pattern here
        return False
    return True


def _register_qconv_weight_prepack_pass(pattern, pass_number):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_conv2d_pattern,
        pass_number=pass_number,
    )
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        int8 activation
          |
        dequant_per_tensor
          |
        Conv2d <- optional(aten.clone.default) <- dequant_per_channel <- int8_weight

        Insert weight prepack node and change the pattern to:
        int8 activation
          |
        onednn.qconv2d_pointwise <- onednn.qconv_prepack <- int8_weight
        """
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        mul_node = conv_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        has_clone_to_channel_last_node_in_pattern = (
            conv_node.args[1].target is aten.clone.default
        )
        clone_node = (
            conv_node.args[1] if has_clone_to_channel_last_node_in_pattern else None
        )
        dequant_per_channel = (
            clone_node.args[0]
            if has_clone_to_channel_last_node_in_pattern
            else conv_node.args[1]
        )
        assert (
            dequant_per_channel.target
            is quantized_decomposed.dequantize_per_channel.default
        )

        # Activation QParams
        qx, x_zp, x_scale = (
            kwargs["x"],
            kwargs["x_zp"],
            kwargs["x_scale"],
        )

        # Weight QParams
        qw, w_scale, w_zp = (
            kwargs["q_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )

        # Conv Params
        bias, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )

        x_shape = qx.meta.get("tensor_meta").shape
        graph = match.graph
        with graph.inserting_before(conv_node):
            # Insert weight prepack node and the QConv node
            packed_weight_inputs = (
                qw,
                w_scale,
                x_scale,
                x_zp,
                stride,
                padding,
                dilation,
                groups,
                x_shape,
            )
            packed_weight_op = torch.ops.onednn.qconv_prepack
            prepack_weight_node = graph.call_function(
                packed_weight_op, args=packed_weight_inputs
            )

            new_args = (
                qx,
                x_scale,
                x_zp,
                prepack_weight_node,
                w_scale,
                w_zp,
                bias,
                stride,
                padding,
                dilation,
                groups,
                1.0,  # inv_output_scale
                0,  # output_zero_point
                True,  # fp32_output
                "none",  # attr
                [],  # scalars
                "",  # algorithm
            )
            new_conv_node = graph.call_function(
                torch.ops.onednn.qconv2d_pointwise.default, args=new_args
            )
            conv_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(conv_node.meta)

            # Erase the original conv node
            graph.erase_node(conv_node)
            # Erase the dequant pattern
            graph.erase_node(mul_node)
            graph.erase_node(sub_node)
            graph.erase_node(to_fp32_node)
            # Erase the dequant per channel pattern
            if clone_node is not None:
                graph.erase_node(clone_node)
            graph.erase_node(dequant_per_channel)


def _generate_dequant_convolution_node_pattern(_dequant_per_channel_pattern):
    dequant_convolution_node_pattern = CallFunction(
        aten.convolution.default,
        dequantize_activation_pattern,
        _dequant_per_channel_pattern,
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("is_transposed"),
        KeywordArg("out_padding"),
        KeywordArg("groups"),
    )
    return dequant_convolution_node_pattern


def _generate_qconv_weight_prepack_patterns():
    return (
        _generate_dequant_convolution_node_pattern(dequant_per_channel_pattern),
        # There is another pattern due to the pass of convert_conv_weights_to_channels_last
        # https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/_inductor/freezing.py#L338-L362.
        # Depend on some heuristics, it may or may not insert to(channel_last) node
        # between convolution and dequant_per_channel node
        _generate_dequant_convolution_node_pattern(
            dequant_per_channel_clone_to_channel_last_pattern
        ),
    )


@functools.lru_cache(None)
def _register_quantization_weight_pack_pass():
    weight_prepack_patterns = _generate_qconv_weight_prepack_patterns()
    for weight_prepack_pattern in weight_prepack_patterns:
        # Register to pass_number 1, so we can do dequant promotion in pass_number 0.
        _register_qconv_weight_prepack_pass(weight_prepack_pattern, pass_number=1)
