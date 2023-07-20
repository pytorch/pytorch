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

dequantize_accum_pattern = CallFunction(
    aten.mul.Tensor,
    CallFunction(
        aten.sub.Tensor,
        CallFunction(
            prims.convert_element_type.default,
            KeywordArg("accum"),
            KeywordArg("accum_dq_dtype"),
        ),
        KeywordArg("accum_zp"),
    ),
    KeywordArg("accum_scale"),
)


def generate_pattern_with_binary(binary_post_op, computation_call, extra_input_pattern):
    return CallFunction(
        binary_post_op,
        computation_call,
        extra_input_pattern,
    )


def generate_pattern_with_unary(computation_call, unary_post_op):
    if unary_post_op is not None:
        return CallFunction(
            unary_post_op,
            computation_call,
        )
    return computation_call


def generate_pattern_with_output_quant(computation_call):
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
                KeywordArg("o_qmin"),
            ),
            KeywordArg("o_qmax"),
        ),
        KeywordArg("o_dtype"),
    )
    return quantize_conv_output_pattern_pt2e


def _register_quantized_conv_lowering(
    pattern,
    pass_number,
    computation_op,
    fp32_output,
    unary_attr,
):
    @register_lowering_pattern(pattern, pass_number=pass_number)
    def qconv(match: Match, *args, **kwargs):
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        # Conv Params
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        # Output QParams
        o_inv_scale, o_zero_point = (
            kwargs["o_inv_scale"],
            kwargs["o_zp"],
        )
        assert (
            kwargs["fp32_output"] is True
        )  # Expected int8-in fp32-out qconv in weight prepack phase
        assert (
            kwargs["attr"] == "none"
        )  # Expected no post op fused in weight prepack phase
        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            fp32_output,
            unary_attr.op_name,
            unary_attr.scalars_attr,
            unary_attr.algorithm_attr,
        )
        return L[computation_op](*computation_args)

    return qconv


def _register_quantized_conv_binary_lowering(
    pattern,
    pass_number,
    computation_op,
    fp32_output,
    binary_unary_attr,
):
    @register_lowering_pattern(pattern, pass_number=pass_number)
    def qconv_binary(match: Match, *args, **kwargs):
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        accum, accum_scale, accum_zp = (
            kwargs["accum"],
            kwargs["accum_scale"],
            kwargs["accum_zp"],
        )
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        o_inv_scale, o_zero_point = (
            kwargs["o_inv_scale"],
            kwargs["o_zp"],
        )

        computation_args = (
            x,
            x_scale,
            x_zp,
            accum,
            accum_scale,
            accum_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            fp32_output,
            binary_unary_attr.binary_op_name,
            binary_unary_attr.alpha,
            binary_unary_attr.unary_op_name,
            binary_unary_attr.scalars_attr,
            binary_unary_attr.algorithm_attr,
        )
        return L[computation_op](*computation_args)

    return qconv_binary


def _register_quantization_unary_fusion():
    class UnaryAttr:
        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    unary_replace_patterns = {
        UnaryAttr("none", [], ""): generate_pattern_with_output_quant(
            dequantize_qconv_pt2e_pattern
        ),
        UnaryAttr("relu", [], ""): generate_pattern_with_output_quant(
            generate_pattern_with_unary(
                dequantize_qconv_pt2e_pattern, aten.relu.default
            )
        ),
    }

    for unary_attr, patterns in unary_replace_patterns.items():
        # Register qconv2d pattern for ExternKernel Lowering
        _register_quantized_conv_lowering(
            patterns,
            1 if unary_attr.op_name != "none" else 2,  # pass_number
            torch.ops.onednn.qconv2d_pointwise,  # computation_op
            False,  # fp32_output
            unary_attr,  # unary_attr
        )


def _register_quantization_binary_fusion():
    class BinaryUnaryAttr:
        def __init__(
            self,
            binary_op_name: str,
            alpha=None,
            unary_op_name: str = "none",
            scalars_attr=None,
            algorithm_attr=None,
        ):
            self.binary_op_name = binary_op_name
            self.alpha = alpha if alpha else 1.0
            self.unary_op_name = unary_op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    binary_replace_patterns = {
        BinaryUnaryAttr("add", 1.0, "none", [], ""): generate_pattern_with_output_quant(
            generate_pattern_with_binary(
                aten.add.Tensor,
                dequantize_qconv_pt2e_pattern,
                dequantize_accum_pattern,
            )
        ),
        BinaryUnaryAttr("add", 1.0, "relu", [], ""): generate_pattern_with_output_quant(
            generate_pattern_with_unary(
                generate_pattern_with_binary(
                    aten.add.Tensor,
                    dequantize_qconv_pt2e_pattern,
                    dequantize_accum_pattern,
                ),
                aten.relu.default,
            )
        ),
    }

    for binary_unary_attr, patterns in binary_replace_patterns.items():
        # Register qconv2d_binary_unary pattern for ExternKernel Lowering
        _register_quantized_conv_binary_lowering(
            patterns,
            0 if binary_unary_attr.unary_op_name != "none" else 1,  # pass_number
            torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
            False,  # fp32_output
            binary_unary_attr,  # binary_unary_attr
        )


def _register_quantization_lowerings():
    _register_quantization_unary_fusion()
    _register_quantization_binary_fusion()


def _is_valid_dequant_promotion_pattern(match):
    mul_node = match.output_node()
    sub_node = mul_node.args[0]
    to_fp32_node = sub_node.args[0]
    if (
        mul_node.target is aten.mul.Tensor
        and sub_node.target is aten.sub.Tensor
        and to_fp32_node.target is prims.convert_element_type.default
        and len(list(mul_node.users)) > 1
    ):
        # dequant pattern has more than 1 users to be promoted
        return True
    return False


def _register_dequant_promotion_pass(pattern, pass_number):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_promotion_pattern,
        pass_number=pass_number,
    )
    def dequant_promotion(match: Match, *args, **kwargs):
        # If dequant pattern used by multiply nodes,
        # we will do dequant promotion. So each user node has a seperate dequant pattern connected.
        def clone_to_new_node(graph, source_node, user_node):
            assert (
                source_node.op == "call_function"
            ), "clone_to_new_node only support node.op call_function"
            with graph.inserting_before(user_node):
                new_node = graph.call_function(
                    source_node.target,
                    args=source_node.args,
                    kwargs=source_node.kwargs,
                )
                new_node.meta = copy.copy(source_node.meta)
                user_node.replace_input_with(source_node, new_node)
            return new_node

        mul_node = match.output_node()
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        assert mul_node.target is aten.mul.Tensor
        assert sub_node.target is aten.sub.Tensor
        assert to_fp32_node.target is prims.convert_element_type.default

        graph = match.graph
        for index in range(len(list(mul_node.users)) - 1):
            user_node = list(mul_node.users)[index]
            # Step1: Duplicate the mul node
            new_mul_node = clone_to_new_node(graph, mul_node, user_node)
            # Step2: Duplicate the sub node
            new_sub_node = clone_to_new_node(graph, sub_node, new_mul_node)
            # Step3: Duplicate the to_fp32 node
            _ = clone_to_new_node(graph, to_fp32_node, new_sub_node)


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
    _register_dequant_promotion_pass(
        dequantize_per_tensor_activation_pattern, pass_number=0
    )  # pass_number=0 to run before weight prepack
    weight_prepack_patterns = _generate_qconv_weight_prepack_patterns()
    for weight_prepack_pattern in weight_prepack_patterns:
        # Register to pass_number 1, so we can do dequant promotion in pass_number 0.
        _register_qconv_weight_prepack_pass(weight_prepack_pattern, pass_number=1)
