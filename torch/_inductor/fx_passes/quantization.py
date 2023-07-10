import functools

import torch
from ..ir import QConv, QConvPointWisePT2E, TensorBox
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


aten_qconv_pt2e_pattern = CallFunction(
    torch.ops.quantized.dynamic_quant_qconv.tensor,
    dequantize_activation_pattern,
    KeywordArg("dynamic_x_scale"),  # x_scale
    KeywordArg("dynamic_x_zp"),  # x_zp
    KeywordArg("packed_weight"),  # packed_weight
    KeywordArg("w_scale"),  # w_scale
    KeywordArg("w_zp"),  # w_zp
    KeywordArg("w_axis"),  # w_axis
    KeywordArg("b"),  # bias
    KeywordArg("stride"),
    KeywordArg("padding"),
    KeywordArg("dilation"),
    KeywordArg("transposed"),
    KeywordArg("o_padding"),
    KeywordArg("groups"),
)

quantize_conv_output_pattern_pt2e = CallFunction(
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
                        aten_qconv_pt2e_pattern,  # output of conv
                        KeywordArg("o_inv_scale"),
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

pattern_match_count = 0


def _register_quantized_conv_lowering_pt2e(pattern):
    @register_lowering_pattern(pattern)
    def qconv(match: Match, *args, **kwargs):
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        b, stride, padding, dilation = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
        )
        groups, o_inv_scale, o_zero_point, o_dtype = (
            kwargs["groups"],
            kwargs["o_inv_scale"],
            kwargs["o_zp"],
            kwargs["o_dtype"],
        )

        # packed_weight = kwargs["packed_weight"]
        packed_weight, w_scale, w_zp, w_axis = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
            kwargs["w_axis"],
        )
        global pattern_match_count
        pattern_match_count += 1
        print(
            "---- matched the pattern ----: {}".format(pattern_match_count), flush=True
        )

        weight_shape = packed_weight.get_size()
        dim = len(weight_shape) - 2
        return TensorBox.create(
            QConvPointWisePT2E.create(
                dim,
                x,
                x_scale,
                x_zp,
                packed_weight,
                w_scale,
                w_zp,
                w_axis,
                b,
                stride,
                padding,
                dilation,
                groups,
                o_inv_scale,
                o_zero_point,
                o_dtype,
                False,  # fp32_output
                "none",  # unary_attr
                [],  # unary_scalars
                "",  # unary_algorithm
            )
        )

    return qconv


def register_quantization_lowerings():
    _register_quantized_conv_lowering(quantize_conv_output_pattern)
    _register_quantized_conv_lowering_pt2e(quantize_conv_output_pattern_pt2e)


dequant_per_channel_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,  # dequant_per_channel node
    KeywordArg("q_weight"),  # bias
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)

dequant_convolution_node_pattern = CallFunction(
    aten.convolution.default,
    dequantize_activation_pattern,
    dequant_per_channel_pattern,
    KeywordArg("b"),  # bias
    KeywordArg("stride"),
    KeywordArg("padding"),
    KeywordArg("dilation"),
    KeywordArg("is_transposed"),
    KeywordArg("out_padding"),
    KeywordArg("groups"),
)


def _register_qconv_weight_prepack_pass(pattern):
    @register_freezing_graph_pattern(
        pattern, pass_number=1
    )  # pass_number=1, ensure it's behand dequant promotion pass
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        to_fp32_node = match.nodes[0]
        sub_node = match.nodes[1]
        mul_node = match.nodes[2]
        dequant_per_channel = match.nodes[3]
        conv_node = match.nodes[4]

        print("---- match dequant weight prepack pattern ----", flush=True)

        bias, stride, padding, dilation, is_transposed, out_padding, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["is_transposed"],
            kwargs["out_padding"],
            kwargs["groups"],
        )

        qx, x_dq_dtype, x_zp, x_scale = (
            kwargs["x"],
            kwargs["x_dq_dtype"],
            kwargs["x_zp"],
            kwargs["x_scale"],
        )

        qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype = (
            kwargs["q_weight"],  # bias
            kwargs["w_scale"],
            kwargs["w_zp"],
            kwargs["w_axis"],
            kwargs["w_quant_min"],
            kwargs["w_quant_max"],
            kwargs["w_dtype"],
        )

        # Use as scale, zp from dequant node to do weight prepack and requant inside dynamic_qconv_op
        x_shape = qx.meta.get("tensor_meta").shape
        graph = match.graph
        with graph.inserting_before(conv_node):
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
            packed_weight_op = torch.ops.quantized.qconv_prepack_pt2e
            prepack_weight_node = graph.call_function(
                packed_weight_op, args=packed_weight_inputs
            )

            new_args = (
                mul_node,
                x_scale,
                x_zp,
                prepack_weight_node,
                w_scale,
                w_zp,
                w_axis,
                bias,
                stride,
                padding,
                dilation,
                is_transposed,
                out_padding,
                groups,
            )
            new_conv_node = graph.call_function(
                torch.ops.quantized.dynamic_quant_qconv.tensor, args=new_args
            )
            conv_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(conv_node.meta)
            graph.erase_node(conv_node)


@functools.lru_cache(None)
def _quantization_mkldnn_weight_pack_init():
    _register_qconv_weight_prepack_pass(dequant_convolution_node_pattern)
