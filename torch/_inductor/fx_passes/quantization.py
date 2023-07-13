import copy
import functools

import torch
from ..lowering import lowerings as L
from ..pattern_matcher import Arg, CallFunction, KeywordArg, Match
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern

aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed

"""
dequantize activation:
    x = x.to(fp32)
    x = x - zero_point
    x = x * scale
"""
dequantize_per_tensor_activation_pattern = CallFunction(
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

dequantize_per_channel_weight_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)


dequantize_per_channel_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

dequantize_qconv_pt2e_pattern = CallFunction(
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
    KeywordArg("inv_output_scale"),  # inv_output_scale = 1.0
    KeywordArg("output_zero_point"),  # output_zero_point = 0
    KeywordArg("fp32_output"),  # fp32_output = True
    KeywordArg("attr"),  # attr = "none"
    Arg(),  # scalars
    Arg(),  # algorithm
)


def generate_pattern_with_output_quant_pattern(computation_call):
    """
    quantize output:
        output = round(output * o_inv_scale)
        output = output + zero_point
        output = clamp_min(output, 0)
        output = clamp_max(output, 127)
        output = output.to(uint8)
    """
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
                            computation_call,
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
    return quantize_conv_output_pattern_pt2e


pattern_match_count = 0


def _register_quantized_conv_lowering(pattern, computation_op, unary_attr):
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
            "---- matched the pattern v2 post op: {0} ----: {1}".format(
                unary_attr,
                pattern_match_count,
            ),
            flush=True,
        )

        assert (
            kwargs["fp32_output"] is True
        )  # Expected int8-in fp32-out qconv in weight prepack phase
        assert (
            kwargs["attr"] == "none"
        )  # Expected no post op fused in weight prepack phase
        weight_shape = packed_weight.get_size()
        dim = len(weight_shape) - 2
        computation_args = (
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
            unary_attr,  # unary_attr
            [],  # unary_scalars
            "",  # unary_algorithm
        )
        return L[computation_op](*computation_args)

    return qconv


def register_quantization_lowerings():
    quantize_conv_output_pattern_pt2e = generate_pattern_with_output_quant_pattern(
        dequantize_qconv_pt2e_pattern
    )
    _register_quantized_conv_lowering(
        quantize_conv_output_pattern_pt2e,
        torch.ops.onednn.qconv2d_pointwise,
        "none",
    )


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


def _is_valid_dequant_promotion_pattern(match):
    to_fp32_node = match.nodes[0]
    sub_node = match.nodes[1]
    mul_node = match.nodes[2]
    assert mul_node.target is torch.ops.aten.mul.Tensor
    # dequant pattern has more than 1 users to be promoted
    return len(list(mul_node.users)) > 1


def _register_dequant_promotion_pass(pattern):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_promotion_pattern,
        pass_number=0,
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


def _is_valid_dequant_conv2d_pattern(match):
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

    (
        to_fp32_node,
        sub_node,
        mul_node,
    ) = match.nodes[0:3]

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
        extra_check=_is_valid_dequant_conv2d_pattern,
        pass_number=1,  # pass_number=1, ensure it's behand dequant promotion pass
    )
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        """
        Macth the pattern:
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
        has_clone_to_channel_last_node_in_pattern = any(
            node.target == aten.clone.default for node in match.nodes
        )

        (
            to_fp32_node,
            sub_node,
            mul_node,
            dequant_per_channel,
        ) = match.nodes[0:4]

        clone_node = (
            match.nodes[4] if has_clone_to_channel_last_node_in_pattern else None
        )
        conv_node = (
            match.nodes[5]
            if has_clone_to_channel_last_node_in_pattern
            else match.nodes[4]
        )
        if clone_node is not None:
            assert clone_node.target is aten.clone.default
        assert conv_node.target is aten.convolution.default

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
        dequantize_per_tensor_activation_pattern,
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
        _generate_dequant_convolution_node_pattern(
            dequantize_per_channel_weight_pattern
        ),
        # There is another pattern due to the pass of convert_conv_weights_to_channels_last
        # https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/_inductor/freezing.py#L338-L362.
        # Depend on some heuristics, it may or may not insert to(channel_last) node
        # between convolution and dequant_per_channel node
        _generate_dequant_convolution_node_pattern(
            dequantize_per_channel_clone_weight_pattern
        ),
    )


@functools.lru_cache(None)
def _register_quantization_weight_pack_pass():
    _register_dequant_promotion_pass(dequant_node_pattern)
    weight_prepack_patterns = _generate_qconv_weight_prepack_patterns()
    for weight_prepack_pattern in weight_prepack_patterns:
        _register_qconv_weight_prepack_pass(weight_prepack_pattern)
