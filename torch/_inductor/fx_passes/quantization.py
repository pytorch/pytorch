import copy
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
    torch.ops.onednn.qconv2d_pointwise.default,
    KeywordArg("x"),
    KeywordArg("x_scale"),  # x_scale
    KeywordArg("x_zp"),  # x_zp
    KeywordArg("packed_weight"),  # packed_weight
    KeywordArg("w_scale"),  # w_scale
    KeywordArg("w_zp"),  # w_zp
    KeywordArg("b"),  # bias
    KeywordArg("stride"),
    KeywordArg("padding"),
    KeywordArg("dilation"),
    KeywordArg("groups"),
    Arg(),  # output_scale
    Arg(),  # output_zero_point
    Arg(),  # fp32_output
    Arg(),  # attr
    Arg(),  # scalars
    Arg(),  # algorithm
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
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
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
                -1,  # w_axis delete it later
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


dequant_node_pattern = CallFunction(
    aten.mul.Tensor,
    CallFunction(
        aten.sub.Tensor,
        CallFunction(
            prims.convert_element_type.default,
            KeywordArg("x"),
            KeywordArg("o_dtype"),  # dtype=torch.float32
        ),
        KeywordArg("dequant_zp"),  # dequant zp
    ),
    KeywordArg("dequant_scale"),  # dequant_scale
)


def _register_dequant_promotion_pass(pattern):
    @register_freezing_graph_pattern(
        pattern, pass_number=0
    )  # pass_number=0, so it will run before insert weight prepack node
    def dequant_promotion(match: Match, *args, **kwargs):
        to_fp32_node = match.nodes[0]
        sub_node = match.nodes[1]
        mul_node = match.nodes[2]
        graph = match.graph
        if len(list(mul_node.users)) > 1:
            # Dequant Node used by multiply nodes
            # Will do dequant promotion, so each used node has a seperate dequant pattern connected
            for index in range(len(list(mul_node.users)) - 1):
                user_node = list(mul_node.users)[index]
                with graph.inserting_before(user_node):
                    # Step1: Duplicate the mul node
                    new_mul_node = graph.call_function(
                        torch.ops.aten.mul.Tensor,
                        args=mul_node.args,
                        kwargs=mul_node.kwargs,
                    )
                    new_mul_node.meta = copy.copy(mul_node.meta)
                    user_node.replace_input_with(mul_node, new_mul_node)

                    with graph.inserting_before(new_mul_node):
                        # Step2: Duplicate the sub node
                        new_sub_node = graph.call_function(
                            torch.ops.aten.sub.Tensor,
                            args=sub_node.args,
                            kwargs=sub_node.kwargs,
                        )
                        new_sub_node.meta = copy.copy(sub_node.meta)
                        new_mul_node.replace_input_with(sub_node, new_sub_node)

                        with graph.inserting_before(new_sub_node):
                            # Step3: Duplicate the to_fp32 node
                            new_to_fp32_node = graph.call_function(
                                torch.ops.prims.convert_element_type.default,
                                args=to_fp32_node.args,
                                kwargs=to_fp32_node.kwargs,
                            )
                            new_to_fp32_node.meta = copy.copy(to_fp32_node.meta)
                            new_sub_node.replace_input_with(
                                to_fp32_node, new_to_fp32_node
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


dequant_per_channel_clone_channel_last_pattern = CallFunction(
    aten.clone.default,
    dequant_per_channel_pattern,
    memory_format=KeywordArg("memory_format"),
)


def _is_dequant_conv2d(match):
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
    to_fp32_node = match.nodes[0]
    sub_node = match.nodes[1]
    mul_node = match.nodes[2]
    assert to_fp32_node.target is torch.ops.prims.convert_element_type.default
    assert sub_node.target is torch.ops.aten.sub.Tensor
    assert mul_node.target is torch.ops.aten.mul.Tensor
    if (
        len(list(to_fp32_node.users)) != 1
        or len(list(sub_node.users)) != 1
        or len(list(mul_node.users)) != 1
    ):
        # Ensure the dequant pattern only has 1 user
        # since we will delete the dequant pattern here
        return False
    return True


def _register_qconv_weight_prepack_pass(pattern):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_dequant_conv2d,
        pass_number=1,  # pass_number=1, ensure it's behand dequant promotion pass
    )
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        print("---- match dequant weight prepack pattern ----", flush=True)

        has_to_channel_last_in_pattern = False
        for node in match.nodes:
            if node.target == aten.clone.default:
                has_to_channel_last_in_pattern = True

        print(
            "has_to_channel_last_in_pattern is: {}".format(
                has_to_channel_last_in_pattern
            ),
            flush=True,
        )

        to_fp32_node = match.nodes[0]
        sub_node = match.nodes[1]
        mul_node = match.nodes[2]
        dequant_per_channel = match.nodes[3]
        clone_node = match.nodes[4] if has_to_channel_last_in_pattern else None
        conv_node = match.nodes[5] if has_to_channel_last_in_pattern else match.nodes[4]

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
                1.0,  # output_scale
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

            if clone_node is not None:
                graph.erase_node(clone_node)

            # Erase the dequant pattern
            graph.erase_node(mul_node)
            graph.erase_node(sub_node)
            graph.erase_node(to_fp32_node)

            # Erase the dequant per channel pattern
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


def _generate_weight_prepack_patterns():
    replacement_weight_prepack_patterns = (
        _generate_dequant_convolution_node_pattern(dequant_per_channel_pattern),
        # There is another pattern due to the pass of convert_conv_weights_to_channels_last
        # https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/_inductor/freezing.py#L338-L362.
        # Depend on some heuristics, it may or may not insert to(channel_last) node
        # between convolution and dequant_per_channel node
        _generate_dequant_convolution_node_pattern(
            dequant_per_channel_clone_channel_last_pattern
        ),
    )
    return replacement_weight_prepack_patterns


@functools.lru_cache(None)
def _quantization_mkldnn_weight_pack_init():
    _register_dequant_promotion_pass(dequant_node_pattern)
    weight_prepack_patterns = _generate_weight_prepack_patterns()
    for weight_prepack_pattern in weight_prepack_patterns:
        _register_qconv_weight_prepack_pass(weight_prepack_pattern)
