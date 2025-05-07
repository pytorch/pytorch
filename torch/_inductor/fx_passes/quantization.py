# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import copy
import functools
import itertools
import math
import operator
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.fx.node import map_arg

from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern


aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized

# Only for per tensor quant since permute may changes the channel idx
_PER_TENSOR_QUANTIZE_OPS = [
    quantized_decomposed.quantize_per_tensor.default,
    quantized_decomposed.quantize_per_tensor.tensor,
]

_VIEW_OPS = [
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
]

"""
The quantization.py file primarily incorporates passes related to quantization fusion
in inductor, includes:
1. Dequant Promotion;
2. Conv/GEMM weight prepack with oneDNN Library;
3. Conv/GEMM quantization fusion with output quant node (if have);
4. Other pointwise operators' quantization fusion like: qmaxpool2d, qcat and more;

It also involves int8-mixed-fp32 and int8-mixed-bf16 quantization. The main difference
of patterns for int8-mixed-bf16, comparing with int8-mixed-fp32, is
1. There is to(dtype=torch.bfloat16) node at the inputs of activation and weight for Conv/GEMM.
2. There is to(dtype=torch.float32) node at the outputs of Conv/GEMM before inputs to next quant node.
Refer to: https://github.com/pytorch/pytorch/issues/111640 for detail design of int8-mixed-bf16
quantization.
"""


def _get_pattern_output_dtype(match: Match):
    """
    Get the pattern's output dtype from node's meta
    Assume only 1 output node in this matched pattern.
    """
    pattern_output_nodes = match.output_nodes()
    assert len(pattern_output_nodes) == 1
    output_node = pattern_output_nodes[0]
    assert isinstance(output_node, torch.fx.Node)
    output_dtype = output_node.meta["val"].dtype
    assert output_dtype in [torch.int8, torch.uint8, torch.float32, torch.bfloat16]
    return output_dtype


def _may_generate_pattern_with_dtype_convert(
    pattern, dtype=Arg(), with_dtype_convert=True, users=1
):
    if with_dtype_convert:
        return CallFunction(
            prims.convert_element_type.default,
            pattern,
            dtype,
            _users=users,
        )
    else:
        return pattern


def _may_generate_pattern_with_reshape(pattern, reshape_size=Arg(), with_reshape=True):
    if with_reshape:
        return CallFunction(
            torch.ops.aten.reshape.default,
            pattern,
            reshape_size,
        )
    else:
        return pattern


def _generate_linear_t_pattern(
    _dequant_per_channel_pattern,
    dtype,
):
    assert dtype in [torch.float32, torch.bfloat16]
    t_pattern = CallFunction(
        aten.permute.default,
        _may_generate_pattern_with_dtype_convert(
            _dequant_per_channel_pattern,
            KeywordArg("autocast_wgt_dtype"),
            dtype == torch.bfloat16,
        ),
        KeywordArg("permute_axes"),
    )
    return t_pattern


def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
    # only insert to_dtype if is_bf16 is True
    computation_call = _may_generate_pattern_with_dtype_convert(
        call_fn, dtype=KeywordArg("to_float"), with_dtype_convert=is_bf16, users=users
    )
    return unary_fusion(computation_call)


def get_dequantize_per_tensor_activation_pattern(is_tensor_overload=False):
    dequantize_per_tensor_activation_pattern = CallFunction(
        quantized_decomposed.dequantize_per_tensor.tensor
        if is_tensor_overload
        else quantized_decomposed.dequantize_per_tensor.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("x_quant_min"),
        KeywordArg("x_quant_max"),
        KeywordArg("x_dq_dtype"),
    )
    return dequantize_per_tensor_activation_pattern


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

dequantize_per_channel_to_bf16_weight_pattern = (
    _may_generate_pattern_with_dtype_convert(
        dequantize_per_channel_weight_pattern,
        KeywordArg("autocast_wgt_dtype"),
    )
)

dequantize_per_channel_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

dequantize_per_channel_to_bf16_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_to_bf16_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)


def get_qconv_pt2e_pattern(users=1):
    return CallFunction(
        torch.ops.onednn.qconv_pointwise.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("postop_name"),
        KeywordArg("postop_args"),
        KeywordArg("postop_algorithm"),
        _users=users,
    )


def get_qconv2d_binary_pt2e_pattern(users=1):
    return CallFunction(
        torch.ops.onednn.qconv2d_pointwise.binary,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("accum"),
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("accum_scale"),
        KeywordArg("accum_zero_point"),
        KeywordArg("binary_op_name"),
        KeywordArg("alpha"),
        KeywordArg("unary_op_name"),
        KeywordArg("unary_op_args"),
        KeywordArg("unary_op_algorithm"),
        _users=users,
    )


def get_qlinear_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.default
    )
    return CallFunction(
        qlinear_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("b"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("postop_name"),
        KeywordArg("postop_args"),
        KeywordArg("postop_algorithm"),
        _users=users,
    )


def get_qlinear_binary_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.binary_tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.binary
    )
    return CallFunction(
        qlinear_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("x_2"),
        KeywordArg("b"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("x2_scale"),
        KeywordArg("x2_zp"),
        KeywordArg("binary_op_name"),
        KeywordArg("alpha"),
        KeywordArg("unary_op_name"),
        KeywordArg("unary_op_args"),
        KeywordArg("unary_op_algorithm"),
        _users=users,
    )


dequantize_accum_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    KeywordArg("accum"),
    KeywordArg("accum_scale"),
    KeywordArg("accum_zp"),
    Arg(),
    Arg(),
    KeywordArg("accum_dq_dtype"),
)


def generate_pattern_with_binary(
    binary_post_op,
    computation_call,
    extra_input_pattern,
    dtype_convert=False,
    swap_inputs=False,
):
    binary_pattern = (
        CallFunction(
            binary_post_op,
            extra_input_pattern,
            computation_call,
        )
        if swap_inputs
        else CallFunction(
            binary_post_op,
            computation_call,
            extra_input_pattern,
        )
    )
    return _may_generate_pattern_with_dtype_convert(
        binary_pattern,
        KeywordArg("convert_dtype_after_inplace_add"),
        dtype_convert,
    )


def generate_pattern_with_unary(computation_call, unary_post_op):
    if unary_post_op is not None:
        return CallFunction(
            unary_post_op,
            computation_call,
        )
    return computation_call


def generate_pattern_with_output_quant(computation_call, with_dtype_convert=False):
    quantized_op_output_pattern_pt2e = CallFunction(
        quantized_decomposed.quantize_per_tensor.default,
        _may_generate_pattern_with_dtype_convert(
            computation_call,
            Arg(),
            with_dtype_convert,
        ),
        KeywordArg("o_inv_scale"),
        KeywordArg("o_zp"),
        KeywordArg("o_qmin"),
        KeywordArg("o_qmax"),
        KeywordArg("o_dtype"),
    )
    return quantized_op_output_pattern_pt2e


def _check_node_kwarg_arg_value(check_node, kwarg_name, args_index, expected_value):
    if kwarg_name in check_node.kwargs:
        actual_value = check_node.kwargs[kwarg_name]
        return actual_value == expected_value
    else:
        assert len(check_node.args) >= (args_index + 1)
        actual_value = check_node.args[args_index]
        return actual_value == expected_value


def _is_valid_quantized_conv_optimization_pattern():
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype in [torch.float32, torch.bfloat16]:
            # Only keep matched pattern with same output_dtype
            qconv_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qconv_pointwise
            )[0]
            return _check_node_kwarg_arg_value(
                qconv_node_after_weight_prepack, "output_dtype", 13, output_dtype
            )
        return True

    return fn


def _is_valid_qconv_post_op_fusion_pattern(has_binary_post_op=False):
    return (
        _is_valid_qconv_binary_optimization_pattern()
        if has_binary_post_op
        else _is_valid_quantized_conv_optimization_pattern()
    )


def _is_valid_qconv_lowering_pattern():
    def fn(match):
        if len(match.nodes) != 1:
            return False
        return match.nodes[0].target in (
            torch.ops.onednn.qconv_pointwise.default,
            torch.ops.onednn.qconv_pointwise.tensor,
            torch.ops.onednn.qconv2d_pointwise.binary,
            torch.ops.onednn.qconv2d_pointwise.binary_tensor,
        )

    return fn


def _register_quantized_conv_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_lowering_pattern(),
        pass_number=pass_number,
    )
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
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype in [torch.int8, torch.uint8, torch.float32, torch.bfloat16]
        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]
        output_dtype = kwargs["output_dtype"]
        # post op
        postop_name = kwargs["postop_name"]
        postop_args = kwargs["postop_args"]
        postop_algorithm = kwargs["postop_algorithm"]

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
            output_dtype,
            postop_name,
            postop_args,
            postop_algorithm,
        )
        counters["inductor"]["qconv_unary_lower_count"] += 1
        counters["inductor"]["qconv_unary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qconv


def _is_valid_quantized_linear_optimization_pattern():
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype in [torch.float32, torch.bfloat16]:
            # Only keep matched pattern with same output_dtype
            qlinear_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qlinear_pointwise
            )[0]
            return _check_node_kwarg_arg_value(
                qlinear_node_after_weight_prepack, "output_dtype", 9, output_dtype
            )
        return True

    return fn


def _is_valid_qlinear_post_op_fusion_pattern(has_binary_post_op=False):
    return (
        _is_valid_qlinear_binary_optimization_pattern()
        if has_binary_post_op
        else _is_valid_quantized_linear_optimization_pattern()
    )


def _is_valid_qlinear_lowering_pattern():
    def fn(match):
        if len(match.nodes) != 1:
            return False
        return match.nodes[0].target in (
            torch.ops.onednn.qlinear_pointwise.default,
            torch.ops.onednn.qlinear_pointwise.tensor,
            torch.ops.onednn.qlinear_pointwise.binary,
            torch.ops.onednn.qlinear_pointwise.binary_tensor,
        )

    return fn


def _register_quantized_linear_unary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_lowering_pattern(),
        pass_number=pass_number,
    )
    def qlinear(match: Match, *args, **kwargs):
        output_dtype = _get_pattern_output_dtype(match)
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

        # bias
        b = kwargs["b"] if "b" in kwargs else None

        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]

        # post op
        postop_name = kwargs["postop_name"]
        postop_args = kwargs["postop_args"]
        postop_algorithm = kwargs["postop_algorithm"]

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            postop_name,
            postop_args,
            postop_algorithm,
        )
        counters["inductor"]["qlinear_unary_lower_count"] += 1
        counters["inductor"]["qlinear_unary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qlinear


def _register_quantized_linear_binary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_lowering_pattern(),
        pass_number=pass_number,
    )
    def qlinear_binary(match: Match, *args, **kwargs):
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype is not None
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        x2 = kwargs["x_2"]
        x2_scale = kwargs["x2_scale"]
        x2_zp = kwargs["x2_zp"]
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        # bias
        b = kwargs["b"] if "b" in kwargs else None
        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]

        x2.realize()
        from .mkldnn_fusion import _can_be_inplace

        binary_op_name = kwargs["binary_op_name"]
        alpha = kwargs["alpha"]
        unary_op_name = kwargs["unary_op_name"]
        unary_op_args = kwargs["unary_op_args"]
        unary_op_algorithm = kwargs["unary_op_algorithm"]

        if binary_op_name == "sum" and not _can_be_inplace(x2):
            # When we enable the GEMM Template, the output of QLinear
            # will be reshaped from 2D back to 3D if the input is 3D.
            # This causes _can_be_inplace(x2) to return False if x2 happens
            # to be the output of QLinear in this scenario.
            # Change the post op from sum to binary add for this case.
            # Refer to test case:
            #   test_mkldnn_pattern_matcher.py::test_qlinear_dequant_promotion_cpu_input_dim_exceeds_2
            binary_op_name = "add"

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            x2,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            x2_scale,
            x2_zp,
            binary_op_name,
            alpha,
            unary_op_name,
            unary_op_args,
            unary_op_algorithm,
        )
        counters["inductor"]["qlinear_binary_lower_count"] += 1
        counters["inductor"]["qlinear_binary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qlinear_binary


def _is_valid_qconv_binary_optimization_pattern():
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qconv_pointwise
    )


def _is_valid_qlinear_binary_optimization_pattern():
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qlinear_pointwise,
        # we don't insert q-dq for extra input due to accuracy issues
        extra_input_from_dequant=False,
    )


def _is_valid_quantized_op_binary_optimization_pattern(
    qop, extra_input_from_dequant=True
):
    # Check if it's a valid Binary Pattern for qconv2d and qlinear:
    # * qop_pointwise should only has one users
    # * If extra_input_from_dequant is True, extra input of binary node should come from dequant pattern
    # * the two inputs of binary node should have attribute "meta" and should be tensors
    # * the two inputs of binary node should have the same shape
    # * All users of the extra input in this pattern should be
    #   ancestor nodes of the compute node, except for the binary node
    #   connected to the compute node.
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)
        compute_node = filter_nodes(match.nodes, qop)[0]
        # qop_pointwise should only have one user
        if len(compute_node.users) != 1:
            return False
        binary_node_inputs = next(iter(compute_node.users)).args
        assert len(binary_node_inputs) == 2, "Expects binary node with 2 inputs"
        if output_dtype in [torch.float32, torch.bfloat16]:
            extra_input_of_binary_node = None
            for arg in binary_node_inputs:
                if arg != compute_node:
                    extra_input_of_binary_node = arg
                    break
            assert extra_input_of_binary_node is not None
            # Extra input of binary node comes from dequant pattern
            if extra_input_from_dequant and (
                (not isinstance(extra_input_of_binary_node, torch.fx.Node))
                or (
                    extra_input_of_binary_node.target
                    != quantized_decomposed.dequantize_per_tensor.default
                )
            ):
                return False

        # the two inputs of binary node should have attribute "meta" and should be tensors
        if not (
            hasattr(binary_node_inputs[0], "meta")
            and isinstance(binary_node_inputs[0].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ) or not (
            hasattr(binary_node_inputs[1], "meta")
            and isinstance(binary_node_inputs[1].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ):
            return False
        # the two inputs of binary node should have the same shape
        if (
            binary_node_inputs[0].meta["val"].size()  # type: ignore[union-attr]
            != binary_node_inputs[1].meta["val"].size()  # type: ignore[union-attr]
        ):
            return False

        # All users of the extra input in this pattern should be
        # ancestor nodes of the compute node, except for the binary node
        # connected to the compute node.

        from .mkldnn_fusion import _get_remaining_users

        extra_input_of_pattern = (
            match.kwargs["other"]
            if "other" in match.kwargs
            else (
                match.kwargs["accum"]
                if (output_dtype in [torch.uint8, torch.int8])
                or (not extra_input_from_dequant)
                else match.kwargs["accum_after_dequant"]
            )
        )
        if (
            len(_get_remaining_users(extra_input_of_pattern, compute_node)) > 1
            or extra_input_of_pattern == compute_node.args[0]
        ):
            return False
        return True

    return fn


def _register_quantized_conv_binary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_lowering_pattern(),
        pass_number=pass_number,
    )
    def qconv_binary(match: Match, *args, **kwargs):
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype is not None
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        accum = kwargs["accum"]
        accum_scale = kwargs["accum_scale"]
        accum_zp = kwargs["accum_zero_point"]
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
        # Output QParams
        output_scale = kwargs["output_scale"]
        output_zero_point = kwargs["output_zero_point"]

        # post ops
        binary_op_name = kwargs["binary_op_name"]
        alpha = kwargs["alpha"]
        unary_op_name = kwargs["unary_op_name"]
        unary_op_args = kwargs["unary_op_args"]
        unary_op_algorithm = kwargs["unary_op_algorithm"]

        accum.realize()
        from .mkldnn_fusion import _can_be_inplace

        assert _can_be_inplace(accum), (
            "QConv Binary Inplace Fusion requires accum is not an alias or mutation."
        )

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            accum,
            b,
            stride,
            padding,
            dilation,
            groups,
            output_scale,
            output_zero_point,
            output_dtype,
            accum_scale,
            accum_zp,
            binary_op_name,
            alpha,
            unary_op_name,
            unary_op_args,
            unary_op_algorithm,
        )
        counters["inductor"]["qconv2d_binary_lower_count"] += 1
        counters["inductor"]["qconv2d_binary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qconv_binary


def _register_quantization_unary_lowering():
    # QConv2d
    for users in [1, 2]:
        qconv_pattern = get_qconv_pt2e_pattern(users)
        _register_quantized_conv_lowering(
            qconv_pattern,
            2,  # pass_number
            torch.ops.onednn.qconv_pointwise.default,  # computation_op
        )

    # QLinear
    for x_scale_zp_are_tensors in (False, True):
        qlinear_pattern = get_qlinear_pt2e_pattern(x_scale_zp_are_tensors)
        computation_op = (
            torch.ops.onednn.qlinear_pointwise.tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.default
        )
        _register_quantized_linear_unary_lowering(
            qlinear_pattern,
            2,  # pass_number
            computation_op,
        )


def _register_quantization_binary_lowering():
    # QConv2d
    for users in (1, 2):
        qconv_pattern = get_qconv2d_binary_pt2e_pattern(users)
        _register_quantized_conv_binary_lowering(
            qconv_pattern,
            2,  # pass_number
            torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
        )

    # QLinear
    for x_scale_zp_are_tensors in (False, True):
        qlinear_pattern = get_qlinear_binary_pt2e_pattern(x_scale_zp_are_tensors)
        computation_op = (
            torch.ops.onednn.qlinear_pointwise.binary_tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.binary
        )
        _register_quantized_linear_binary_lowering(
            qlinear_pattern,
            2,  # pass_number
            computation_op,
        )


def _is_valid_quantized_maxpool2d_optimization_pattern():
    def fn(match):
        # Only match the pattern which max_pool2d_with_indices returns value
        # instead of indices.
        get_item_node = filter_nodes(match.nodes, operator.getitem)[0]
        return get_item_node.args[1] == 0

    return fn


def _register_quantized_maxpool2d_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_quantized_maxpool2d_optimization_pattern(),
    )
    def qmaxpool2d(match: Match, *args, **kwargs):
        x = kwargs["x"]
        kernel_size = kwargs["kernel_size"]
        stride = kwargs["stride"] if ("stride" in kwargs) else None
        padding = kwargs["padding"] if ("padding" in kwargs) else 0
        dilation = kwargs["dilation"] if ("dilation" in kwargs) else 1
        ceil_mode = kwargs["ceil_mode"] if ("ceil_mode" in kwargs) else False

        if padding == 0:
            padding = [0, 0]
        if dilation == 1:
            dilation = [1, 1]
        if not stride:
            stride = kernel_size
        kernel_size = pad_listlike(kernel_size, 2)
        stride = pad_listlike(stride, 2)
        padding = pad_listlike(padding, 2)
        dilation = pad_listlike(dilation, 2)

        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        assert len(dilation) == 2

        computation_args = (
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        computation_args, _ = require_channels_last(computation_op, *computation_args)
        counters["inductor"]["qmaxpool2d_matcher_count"] += 1
        counters["inductor"]["qmaxpool2d_matcher_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qmaxpool2d


def _register_quantization_maxpool2d():
    # Currently, the default parameters are not in FX Graph generated by Dynamo export.
    # So, if user defines nn.MaxPool2d with different assignment of default parameter,
    # it will generate graph with different number of input nodes and hence
    # different pattern to be matched.
    # Refer to the issue: https://github.com/pytorch/pytorch/issues/105901
    max_pool2d_args_list = [
        [
            KeywordArg("stride"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
            KeywordArg("ceil_mode"),
        ],
    ]
    for max_pool2d_args in max_pool2d_args_list:
        dequantize_maxpool2d_pattern = CallFunction(
            aten.max_pool2d_with_indices.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
        )
        dequantize_lowmem_maxpool2d_pattern = CallFunction(
            prims._low_memory_max_pool_with_offsets.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
            KeywordArg("offset_dtype"),
        )
        dequantize_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_maxpool2d_pattern,
            Arg(),
        )
        dequantize_lowmem_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_lowmem_maxpool2d_pattern,
            Arg(),
        )
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(dequantize_maxpool2d_get_item_pattern),
            quantized.max_pool2d.default,
        )
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(
                dequantize_lowmem_maxpool2d_get_item_pattern
            ),
            quantized.max_pool2d.default,
        )


def _is_input_output_same_scale_zp(check_node):
    def fn(match):
        # Ensure all the inputs and output has same scale and zero point
        # Step 1: Check inputs/output zero point
        # Get dequant nodes at input
        dequant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.dequantize_per_tensor.default
        )
        zero_points = [node.args[2] for node in dequant_nodes]
        # Get quant nodes at output
        quant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.quantize_per_tensor.default
        )
        assert len(quant_nodes) == 1, "expect only 1 add node at output quant pattern"
        zero_points.append(quant_nodes[0].args[2])
        if not all(zero_point == zero_points[0] for zero_point in zero_points):
            return False

        # Step 2: Check inputs/output scale
        scales = [node.args[1] for node in dequant_nodes]
        scales.append(quant_nodes[0].args[1])
        if not all(math.isclose(scale, scales[0], rel_tol=1e-5) for scale in scales):  # type: ignore[arg-type]
            return False

        return True

    return fn


def _register_quantized_cat_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.cat.default),
    )
    def qcat(match: Match, inputs, dim, **kwargs):
        # inputs is with format: [[x1, x1_dq_dtype, x1_zp, x1_scale], ...]
        uint8_inputs = [input[0] for input in inputs]
        counters["inductor"]["qcat_matcher_count"] += 1
        counters["inductor"]["qcat_matcher_nodes"] += len(match.nodes)
        return L[computation_op](uint8_inputs, dim)

    return qcat


_raw_dequantize_per_tensor_activation_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
)


def _register_quantization_cat():
    dequantize_cat_pattern = CallFunction(
        aten.cat.default,
        ListOf(_raw_dequantize_per_tensor_activation_pattern),
        KeywordArg("dim"),
    )
    _register_quantized_cat_lowering(
        generate_pattern_with_output_quant(dequantize_cat_pattern),
        aten.cat,
    )


def _register_quantized_reshape_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.reshape.default),
    )
    def qreshape(match: Match, *args, **kwargs):
        qx = kwargs["x"]
        shape = kwargs["shape"]
        counters["inductor"]["qreshape_matcher_count"] += 1
        counters["inductor"]["qreshape_matcher_nodes"] += len(match.nodes)
        return L[computation_op](qx, shape)

    return qreshape


def _register_quantization_reshape():
    dequantize_reshape_pattern = CallFunction(
        torch.ops.aten.reshape.default,
        get_dequantize_per_tensor_activation_pattern(),
        KeywordArg("shape"),
    )
    _register_quantized_reshape_lowering(
        generate_pattern_with_output_quant(dequantize_reshape_pattern),
        aten.reshape,
    )


def _is_valid_woq_optimization_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("x", "weight", "scales"))
        if not all(
            hasattr(match.kwargs[key], "meta") for key in ["x", "weight", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        weight = match.kwargs["weight"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and weight.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            # _weight_int8pack_mm kernel only supports cpu now
            # TODO: add cuda kernel support instead of calling mul+sum
            and x.device.type == "cpu"
            and x.device == weight.device
            and x.device == scales.device
        )

    return fn


def _register_woq_lowering(pattern, computation_woq, computation_reshape):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_woq_optimization_pattern(),
    )
    def woq(match: Match, *args, **kwargs):
        x = kwargs["x"]
        weight = kwargs["weight"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = weight.get_size()[0]
        origin_x_size = x.get_size()
        x_shape = [-1, origin_x_size[-1]]
        out_shape = origin_x_size[:-1] + [
            out_features,
        ]
        func1 = L[computation_reshape](x, x_shape)
        func2 = L[computation_woq](func1, weight, scales)
        return L[computation_reshape](func2, out_shape)

    return woq


def _register_woq_mm_int8_pattern1():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, with x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                CallFunction(aten.reshape.default, KeywordArg("x"), Arg()),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern2():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, w/o x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                KeywordArg("x"),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern3():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to bmm
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.bmm.default,
            CallFunction(aten.expand.default, KeywordArg("x"), Arg()),
            CallFunction(
                aten.expand.default,
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern4():
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.mm.default,
            KeywordArg("x"),
            CallFunction(
                prims.convert_element_type.default,
                CallFunction(
                    aten.permute.default,
                    KeywordArg("weight"),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_quantization_lowerings():
    _register_quantization_unary_lowering()
    _register_quantization_binary_lowering()
    _register_quantization_maxpool2d()
    _register_quantization_cat()
    _register_quantization_reshape()


def _register_woq_lowerings():
    _register_woq_mm_int8_pattern1()
    _register_woq_mm_int8_pattern2()
    _register_woq_mm_int8_pattern3()
    _register_woq_mm_int8_pattern4()


def _is_valid_dequant_promotion_pattern(dtype=torch.float32):
    def _inner(match):
        assert dtype in [torch.float32, torch.bfloat16]
        dequant_pattern_end_node = match.output_node()
        if dequant_pattern_end_node.target not in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
            prims.convert_element_type.default,
            aten.reshape.default,
        ]:
            return False

        if dequant_pattern_end_node.target is aten.reshape.default:
            dequant_node = (
                dequant_pattern_end_node.args[
                    0
                ]  # pattern: linear <- reshape <- dequant
                if dtype == torch.float32
                else dequant_pattern_end_node.args[0].args[
                    0
                ]  # pattern: linear <- reshape <- to_bf16 <- dequant
            )
        else:
            dequant_node = (
                dequant_pattern_end_node  # pattern: linear <- dequant
                if dtype == torch.float32
                else dequant_pattern_end_node.args[
                    0
                ]  # pattern: linear <- to_bf16 <- dequant
            )

        if (
            dequant_node.target
            in [
                quantized_decomposed.dequantize_per_tensor.default,
                quantized_decomposed.dequantize_per_tensor.tensor,
            ]
            and len(list(dequant_pattern_end_node.users)) > 1
        ):
            # If dequant pattern has more than 1 users, then do dequant promoted
            return True
        return False

    return _inner


def _register_dequant_promotion_pass(pattern, pass_number, dtype=torch.float32):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_promotion_pattern(dtype),
        pass_number=pass_number,
    )
    def dequant_promotion(match: Match, *args, **kwargs):
        # Dequant_promotion will transform
        # graph 1:
        #            quant
        #      + - - - | - - - +
        #      |    dequant    |
        #      |    /     \    |
        #      |  node1  node2 |
        #      + - | - - - | - +
        #        quant   quant
        # into:
        # graph 2:
        #            quant
        #      + - - / - \ - - +
        #      |dequant dequant|
        #      |    |      |   |
        #      | node1 node2   |
        #      + - | - - - | - +
        #        quant   quant
        # In graph 1, the dequant node is shared by node1 and node2,
        # as a result, neither node1 nor node2 could form an int8
        # fusion pattern.
        # After this transformation, the graph 2 could hit the int8
        # fusion pattern: dequant-node-quant, respectively for
        # node1 and node2.
        assert dtype in [torch.float32, torch.bfloat16]

        def clone_to_new_node(graph, source_node, user_node):
            # Clone the source_node to a new node
            # Replace user_node's input from source_node to new_node
            assert source_node.op == "call_function", (
                "clone_to_new_node only support node.op call_function"
            )
            with graph.inserting_before(user_node):
                new_node = graph.call_function(
                    source_node.target,
                    args=source_node.args,
                    kwargs=source_node.kwargs,
                )
                new_node.meta = copy.copy(source_node.meta)
                user_node.replace_input_with(source_node, new_node)
            return new_node

        # Find the start node and end node of a dequant pattern
        # * End node should be the match.output_node()
        # * Start node should be the node of dequantize_per_tensor
        dequant_pattern_end_node = match.output_node()
        assert dequant_pattern_end_node.target in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
            prims.convert_element_type.default,
            aten.reshape.default,
        ]

        # For a dequant pattern, we should expect see the node list as:
        # * OPT(aten.reshape.default)
        # * OPT(prims.convert_element_type.default) (to_bf16)
        # * dequantize_per_tensor
        def _find_first_node_in_dequant_pattern(_node):
            if _node.target in [
                quantized_decomposed.dequantize_per_tensor.default,
                quantized_decomposed.dequantize_per_tensor.tensor,
            ]:
                # For a dequant pattern, we expect the start node is a dequantize_per_tensor node
                return _node
            else:
                assert len(_node.args) >= 1, (
                    "In in dequant pattern, each node should have more than 1 arg."
                )
                return _find_first_node_in_dequant_pattern(_node.args[0])

        dequant_pattern_start_node = _find_first_node_in_dequant_pattern(
            dequant_pattern_end_node
        )

        assert dequant_pattern_start_node.target in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
        ]

        # Clone the dequant pattern for each user node
        graph = match.graph
        user_node_list = list(dequant_pattern_end_node.users)
        for user_node in user_node_list[1:]:
            _source_node = dequant_pattern_end_node
            _user_node = user_node
            while _source_node != dequant_pattern_start_node.args[0]:
                _user_node = clone_to_new_node(graph, _source_node, _user_node)
                _source_node = _source_node.args[0]  # type: ignore[assignment]

        counters["inductor"]["dequant_promotion_matcher_count"] += 1
        counters["inductor"]["dequant_promotion_matcher_nodes"] += len(match.nodes)


def _is_valid_dequant_conv_pattern(dtype):
    def _inner(match):
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
                or (meta_value.device.type != "cpu" and meta_value.device.type != "xpu")
                or meta_value.dim() not in [3, 4]
            ):
                # Only support conv1d/2d now
                return False

        assert dtype in [torch.float32, torch.bfloat16]

        if dtype == torch.float32:
            dequant_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            dequant_node = convert_to_bf16.args[0]

        if len(list(dequant_node.users)) != 1:
            # Ensure the dequant pattern only has 1 user
            # since we will delete the dequant pattern here
            return False
        return True

    return _inner


def _register_qconv_weight_prepack_pass(pattern, pass_number, dtype=torch.float32):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_conv_pattern(dtype),
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
        onednn.qconv_pointwise <- onednn.qconv_prepack <- int8_weight
        """
        assert dtype in [torch.float32, torch.bfloat16]
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        if dtype == torch.float32:
            dequant_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            dequant_node = convert_to_bf16.args[0]  # type: ignore[union-attr]
        has_clone_to_channel_last_node_in_pattern = (
            conv_node.args[1].target is aten.clone.default  # type: ignore[union-attr]
        )
        clone_node = (
            conv_node.args[1] if has_clone_to_channel_last_node_in_pattern else None
        )

        if dtype == torch.float32:
            dequant_per_channel = (
                clone_node.args[0]  # type: ignore[union-attr]
                if has_clone_to_channel_last_node_in_pattern
                else conv_node.args[1]
            )
        else:
            weight_to_bf16_node = (
                clone_node.args[0]  # type: ignore[union-attr]
                if has_clone_to_channel_last_node_in_pattern
                else conv_node.args[1]
            )
            dequant_per_channel = weight_to_bf16_node.args[0]  # type: ignore[union-attr]

        assert (
            dequant_per_channel.target  # type: ignore[union-attr]
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
        if has_free_symbols(x_shape):
            # For dynamic shape case, we can't get activation shape ahead of runtime.
            x_shape = None
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

            new_args: tuple[Any, ...] = (
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
                dtype,  # output_dtype
                "none",  # attr
                [],  # scalars
                "",  # algorithm
            )
            new_conv_node = graph.call_function(
                torch.ops.onednn.qconv_pointwise.default, args=new_args
            )
            conv_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(conv_node.meta)

            # Erase the original conv node
            graph.erase_node(conv_node)
            # Erase the dequant pattern
            if dtype == torch.bfloat16:
                graph.erase_node(convert_to_bf16)  # type: ignore[possibly-undefined, arg-type]
            graph.erase_node(dequant_node)  # type: ignore[arg-type]
            # Erase the dequant per channel pattern
            if clone_node is not None:
                graph.erase_node(clone_node)  # type: ignore[arg-type]
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)  # type: ignore[possibly-undefined, arg-type]
            graph.erase_node(dequant_per_channel)  # type: ignore[arg-type]
            counters["inductor"]["qconv_weight_prepack_matcher_count"] += 1
            counters["inductor"]["qconv_weight_prepack_matcher_nodes"] += len(
                match.nodes
            )


def _generate_dequant_convolution_node_pattern(
    _dequant_per_channel_pattern, dtype=torch.float32
):
    assert dtype in [torch.float32, torch.bfloat16]
    dequant_convolution_node_pattern = CallFunction(
        aten.convolution.default,
        _may_generate_pattern_with_dtype_convert(
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("autocast_act_dtype"),
            dtype == torch.bfloat16,
        ),
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


def _generate_qconv_weight_prepack_patterns(dtype=torch.float32):
    assert dtype in [torch.float32, torch.bfloat16]
    return (
        _generate_dequant_convolution_node_pattern(
            dequantize_per_channel_weight_pattern
            if dtype == torch.float32
            else dequantize_per_channel_to_bf16_weight_pattern,
            dtype,
        ),
        # There is another pattern due to the pass of convert_conv_weights_to_channels_last
        # https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/_inductor/freezing.py#L338-L362.
        # Depend on some heuristics, it may or may not insert to(channel_last) node
        # between convolution and dequant_per_channel node
        _generate_dequant_convolution_node_pattern(
            dequantize_per_channel_clone_weight_pattern
            if dtype == torch.float32
            else dequantize_per_channel_to_bf16_clone_weight_pattern,
            dtype,
        ),
    )


def _get_linear_node(match, input_dim_exceeds_two, input_contiguous):
    output_reshape_node = None
    if input_dim_exceeds_two:
        if input_contiguous:
            output_reshape_node = match.output_node()
            assert output_reshape_node.target is aten.reshape.default
            linear_node = output_reshape_node.args[0]
        else:
            linear_nodes = filter_nodes(match.nodes, aten.bmm.default)
            assert len(linear_nodes) == 1
            linear_node = linear_nodes[0]
    else:
        linear_node = match.output_node()

    assert linear_node.target in (
        aten.addmm.default,
        aten.mm.default,
        aten.bmm.default,
    )
    return linear_node, output_reshape_node


def _get_linear_dq_node(
    linear_node, input_index, dtype, input_dim_exceeds_two, input_contiguous
):
    act_reshape_node = None
    activation_to_bf16_node = None
    act_expand_node = None
    if input_dim_exceeds_two:
        if input_contiguous:
            act_reshape_node = linear_node.args[input_index]
            assert act_reshape_node.target is aten.reshape.default
            if dtype == torch.float32:
                # pattern: linear -> reshape -> dequant
                dequant_node = act_reshape_node.args[0]
            else:
                # pattern: linear -> reshape -> to_bf16 -> dequant
                activation_to_bf16_node = act_reshape_node.args[0]
                dequant_node = activation_to_bf16_node.args[0]
        else:
            # bmm pattern decomposed from linear when input dim exceeds 2 and not contiguous
            act_expand_node = linear_node.args[input_index]
            assert act_expand_node.target is aten.expand.default
            if dtype == torch.float32:
                dequant_node = act_expand_node.args[0]
            else:
                activation_to_bf16_node = act_expand_node.args[0]
                dequant_node = activation_to_bf16_node.args[0]
    else:
        if dtype == torch.float32:
            # pattern: linear -> dequant
            dequant_node = linear_node.args[input_index]
        else:
            # pattern: linear -> to_bf16 -> dequant
            activation_to_bf16_node = linear_node.args[input_index]
            dequant_node = activation_to_bf16_node.args[0]
    return dequant_node, act_reshape_node, activation_to_bf16_node, act_expand_node


def _is_valid_dequant_linear_pattern(dtype, input_dim_exceeds_two, input_contiguous):
    def _inner(match):
        # Check dequant pattern has only 1 user.
        (
            linear_node,
            _,
        ) = _get_linear_node(match, input_dim_exceeds_two, input_contiguous)

        input_index = 1 if linear_node.target is aten.addmm.default else 0
        assert dtype in [torch.float32, torch.bfloat16]
        (
            dequant_node,
            _,
            _,
            _,
        ) = _get_linear_dq_node(
            linear_node, input_index, dtype, input_dim_exceeds_two, input_contiguous
        )

        assert dequant_node.target in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
        ]

        if len(list(dequant_node.users)) != 1:
            # Ensure the dequant pattern only has 1 user
            # since we will delete the dequant pattern here
            return False

        # Extra check for bmm pattern
        if input_dim_exceeds_two and not input_contiguous:
            # Check for act
            # Act expand size should be exactly same as act size
            act_expand_size = match.kwargs["act_expand_size"]
            act_node = match.kwargs["x"]
            if not (
                hasattr(act_node, "meta")
                and isinstance(act_node.meta.get("val", None), torch.Tensor)
                and (act_node.meta["val"].size() == torch.Size(act_expand_size))
            ):
                return False

            # Check for wgt
            # wgt permute dims should be [1, 0]
            wgt_permute_dims = match.kwargs["permute_axes"]
            if wgt_permute_dims != [1, 0]:
                return False

            # Check below wgt size items:
            # wgt before expand should with dim 2
            # Expand size should with dim 3
            # Expand size[0] should same as act size[0]
            # Expand size[1] should same as wgt size[1]
            # Expand size[2] should same as wgt size[0]
            qweight_node = match.kwargs["q_weight"]
            wgt_expand_size = match.kwargs["wgt_expand_size"]
            if not (
                hasattr(qweight_node, "meta")
                and isinstance(qweight_node.meta.get("val", None), torch.Tensor)
                and len(qweight_node.meta["val"].size()) == 2
                and len(wgt_expand_size) == 3
                and wgt_expand_size[0] == act_node.meta["val"].size()[0]
                and wgt_expand_size[1] == qweight_node.meta["val"].size()[1]
                and wgt_expand_size[2] == qweight_node.meta["val"].size()[0]
            ):
                return False

        return True

    return _inner


def _register_qlinear_weight_prepack_pass(
    pattern,
    pass_number,
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    input_contiguous=True,
):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_linear_pattern(
            dtype, input_dim_exceeds_two, input_contiguous
        ),
        pass_number=pass_number,
    )
    def qlinear_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        int8 activation
          |
        dequant_per_tensor
          |
        mm/addmm <- t <- dequant_per_channel <- int8_weight

        Insert weight prepack node and change the pattern to:
        int8 activation
          |
        onednn.qlinear_pointwise <- onednn.qlinear_prepack <- int8_weight
        """
        assert dtype in [torch.float32, torch.bfloat16]
        (
            linear_node,
            output_reshape_node,
        ) = _get_linear_node(match, input_dim_exceeds_two, input_contiguous)
        input_index = 1 if linear_node.target is aten.addmm.default else 0
        weight_index = input_index + 1

        (
            dequant_node,
            act_reshape_node,
            activation_to_bf16_node,
            act_expand_node,
        ) = _get_linear_dq_node(
            linear_node, input_index, dtype, input_dim_exceeds_two, input_contiguous
        )

        if input_dim_exceeds_two and not input_contiguous:
            wgt_expand_node = linear_node.args[weight_index]
            assert wgt_expand_node.target is aten.expand.default
            t_node = wgt_expand_node.args[0]
        else:
            t_node = linear_node.args[weight_index]

        if dtype == torch.float32:
            dequant_per_channel = t_node.args[0]
        else:
            weight_to_bf16_node = t_node.args[0]
            dequant_per_channel = weight_to_bf16_node.args[0]
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

        # Params
        bias = kwargs["b"] if "b" in kwargs else None

        x_shape = qx.meta.get("tensor_meta").shape
        if has_free_symbols(x_shape):
            # For dynamic shape case, we can't get activation shape ahead of runtime.
            x_shape = None
        graph = match.graph
        with graph.inserting_before(linear_node):
            # Insert weight prepack node and the qlinear node
            packed_weight_inputs = (
                qw,
                x_shape,
            )
            packed_weight_op = torch.ops.onednn.qlinear_prepack
            prepack_weight_node = graph.call_function(
                packed_weight_op, args=packed_weight_inputs
            )

            new_args: tuple[Any, ...] = (
                qx,
                x_scale,
                x_zp,
                prepack_weight_node,
                w_scale,
                w_zp,
                bias,
                1.0,  # output_scale
                0,  # output_zero_point
                dtype,  # output_dtype
                "none",  # post op name
                [],  # post op args
                "",  # post op algorithm
            )
            Node = torch.fx.node.Node
            if isinstance(x_scale, Node) and isinstance(x_zp, Node):
                new_linear_node = graph.call_function(
                    torch.ops.onednn.qlinear_pointwise.tensor, args=new_args
                )
            else:
                new_linear_node = graph.call_function(
                    torch.ops.onednn.qlinear_pointwise.default, args=new_args
                )
            if input_dim_exceeds_two:
                if input_contiguous:
                    output_reshape_node.replace_all_uses_with(new_linear_node)
                    new_linear_node.meta.update(output_reshape_node.meta)
                else:
                    if bias:
                        output_add_node_for_bias = match.output_node()
                        assert output_add_node_for_bias.target is aten.add.Tensor
                        output_add_node_for_bias.replace_all_uses_with(new_linear_node)
                        new_linear_node.meta.update(output_add_node_for_bias.meta)
                    else:
                        linear_node.replace_all_uses_with(new_linear_node)
                        new_linear_node.meta.update(linear_node.meta)
            else:
                linear_node.replace_all_uses_with(new_linear_node)
                new_linear_node.meta.update(linear_node.meta)

            # Erase the original linear node
            if input_dim_exceeds_two:
                if input_contiguous:
                    graph.erase_node(output_reshape_node)
                elif not input_contiguous and bias:
                    graph.erase_node(output_add_node_for_bias)  # type: ignore[possibly-undefined]
            graph.erase_node(linear_node)
            if input_dim_exceeds_two:
                if input_contiguous:
                    graph.erase_node(act_reshape_node)
                else:
                    graph.erase_node(act_expand_node)
                    graph.erase_node(wgt_expand_node)  # type: ignore[possibly-undefined]
            if dtype == torch.bfloat16:
                graph.erase_node(activation_to_bf16_node)
            # Erase the dequant pattern
            graph.erase_node(dequant_node)
            # Erase the dequant per channel pattern
            graph.erase_node(t_node)
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)  # type: ignore[possibly-undefined]
            graph.erase_node(dequant_per_channel)

            counters["inductor"]["qlinear_weight_prepack_matcher_count"] += 1
            counters["inductor"]["qlinear_weight_prepack_matcher_nodes"] += len(
                match.nodes
            )


def _generate_dequant_linear_node_pattern(
    _dequant_per_channel_pattern,
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    is_tensor_overload=False,
):
    assert dtype in [torch.float32, torch.bfloat16]
    t_pattern = _generate_linear_t_pattern(_dequant_per_channel_pattern, dtype)
    dequant_linear_bias_pattern = _may_generate_pattern_with_reshape(
        CallFunction(
            aten.addmm.default,
            KeywordArg("b"),
            _may_generate_pattern_with_reshape(
                _may_generate_pattern_with_dtype_convert(
                    get_dequantize_per_tensor_activation_pattern(is_tensor_overload),
                    KeywordArg("autocast_act_dtype"),
                    dtype == torch.bfloat16,
                ),
                KeywordArg("act_reshape_size"),
                input_dim_exceeds_two,
            ),
            t_pattern,
        ),
        KeywordArg("output_reshape_size"),
        input_dim_exceeds_two,
    )
    dequant_linear_no_bias_pattern = _may_generate_pattern_with_reshape(
        CallFunction(
            aten.mm.default,
            _may_generate_pattern_with_reshape(
                _may_generate_pattern_with_dtype_convert(
                    get_dequantize_per_tensor_activation_pattern(is_tensor_overload),
                    KeywordArg("autocast_act_dtype"),
                    dtype == torch.bfloat16,
                ),
                KeywordArg("act_reshape_size"),
                input_dim_exceeds_two,
            ),
            t_pattern,
        ),
        KeywordArg("output_reshape_size"),
        input_dim_exceeds_two,
    )
    return dequant_linear_bias_pattern, dequant_linear_no_bias_pattern


def _generate_dequant_bmm_node_pattern(
    _dequant_per_channel_pattern,
    dtype=torch.float32,
    with_bias=False,
    is_tensor_overload=False,
):
    # When activation of linear dim exceed 2 and not contiguous
    t_pattern = _generate_linear_t_pattern(_dequant_per_channel_pattern, dtype)

    assert dtype in [torch.float32, torch.bfloat16]
    dequant_bmm_pattern = CallFunction(
        aten.bmm.default,
        CallFunction(
            aten.expand.default,
            _may_generate_pattern_with_dtype_convert(
                get_dequantize_per_tensor_activation_pattern(is_tensor_overload),
                KeywordArg("autocast_act_dtype"),
                dtype == torch.bfloat16,
            ),
            KeywordArg("act_expand_size"),
        ),
        CallFunction(
            aten.expand.default,
            t_pattern,
            KeywordArg("wgt_expand_size"),
        ),
    )

    def _generate_pattern_with_output_add(_dequant_bmm_pattern, _with_bias):
        if _with_bias:
            return CallFunction(
                aten.add.Tensor,
                _dequant_bmm_pattern,
                KeywordArg("b"),
            )
        else:
            return _dequant_bmm_pattern

    return _generate_pattern_with_output_add(dequant_bmm_pattern, with_bias)


def _generate_qlinear_weight_prepack_patterns(
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    input_contiguous=True,
    with_bias=False,
    is_tensor_overload=False,
):
    if input_dim_exceeds_two and not input_contiguous:
        return _generate_dequant_bmm_node_pattern(
            dequantize_per_channel_weight_pattern,
            dtype,
            with_bias,
            is_tensor_overload,
        )
    else:
        return _generate_dequant_linear_node_pattern(
            dequantize_per_channel_weight_pattern,
            dtype,
            input_dim_exceeds_two,
            is_tensor_overload,
        )


def _generate_linear_dynamic_fp16_pattern(
    _dequant_weight_pattern,
    input_dim_exceeds_two=False,
    input_contiguous=True,
    relu_fused=False,
):
    dtype = torch.float32
    t_pattern = _generate_linear_t_pattern(_dequant_weight_pattern, dtype)

    if input_dim_exceeds_two and not input_contiguous:
        # pattern is
        #                   x -> expand -> bmm (-> add) (-> relu)
        # w -> dequant -> permute -> expand /
        pattern_no_bias = CallFunction(
            aten.bmm.default,
            CallFunction(
                aten.expand.default,
                KeywordArg("x"),
                KeywordArg("act_expand_size"),
            ),
            CallFunction(
                aten.expand.default,
                t_pattern,
                KeywordArg("wgt_expand_size"),
            ),
        )
        pattern_with_bias = CallFunction(
            aten.add.Tensor,
            pattern_no_bias,
            KeywordArg("b"),
        )
        if relu_fused:
            pattern_with_bias = CallFunction(aten.relu.default, pattern_with_bias)
            pattern_no_bias = CallFunction(aten.relu.default, pattern_no_bias)
        return pattern_with_bias, pattern_no_bias

    x_pattern_with_reshape = _may_generate_pattern_with_reshape(
        KeywordArg("x"),
        KeywordArg("act_reshape_size"),
        input_dim_exceeds_two,
    )
    dequant_linear_bias_pattern = generate_pattern_with_unary(
        _may_generate_pattern_with_reshape(
            CallFunction(
                aten.addmm.default,
                KeywordArg("b"),
                x_pattern_with_reshape,
                t_pattern,
            ),
            KeywordArg("output_reshape_size"),
            input_dim_exceeds_two,
        ),
        aten.relu.default if relu_fused else None,
    )
    dequant_linear_no_bias_pattern = generate_pattern_with_unary(
        _may_generate_pattern_with_reshape(
            CallFunction(
                aten.mm.default,
                x_pattern_with_reshape,
                t_pattern,
            ),
            KeywordArg("output_reshape_size"),
            input_dim_exceeds_two,
        ),
        aten.relu.default if relu_fused else None,
    )
    return dequant_linear_bias_pattern, dequant_linear_no_bias_pattern


def _register_dequant_promotion():
    dequant_pattern_cases = itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    )
    for dtype, input_dim_exceeds_two, is_tensor_overload in dequant_pattern_cases:
        # 4 dequantization patterns will be matched based on the dtype and input dimension size.
        # Case 1: int8-mixed-fp32, input dim size is 2
        # Case 2: int8-mixed-fp32, input dim size exceeds 2
        # Case 3: int8-mixed-bf16, input dim size is 2
        # Case 4: int8-mixed-bf16, input dim size exceeds 2
        #           quant
        #   + - - - - | - - - - +
        #   |      dequant      |
        #   |         |         |
        #   |    OPT(to_bf16)   |
        #   |         |         |
        #   |    OPT(reshape)   |
        #   |      /     \      |
        #   |    node1  node2   |
        #   + - - | - - - | - - +
        #  OPT(reshape) OPT(reshape)
        #   + - - | - - - | - - +
        #  OPT(to_fp32) OPT(to_fp32)
        #   + - - | - - - | - - +
        #       quant   quant
        _register_dequant_promotion_pass(
            _may_generate_pattern_with_reshape(
                _may_generate_pattern_with_dtype_convert(
                    get_dequantize_per_tensor_activation_pattern(
                        is_tensor_overload=is_tensor_overload
                    ),
                    KeywordArg("autocast_act_dtype"),
                    dtype == torch.bfloat16,
                ),
                KeywordArg("act_reshape_size"),
                with_reshape=input_dim_exceeds_two,
            ),
            pass_number=0,
            dtype=dtype,
        )  # pass_number=0 to run before weight prepack


def _register_qconv_weight_prepack():
    for dtype in [torch.float32, torch.bfloat16]:
        weight_prepack_patterns = _generate_qconv_weight_prepack_patterns(dtype)
        for weight_prepack_pattern in weight_prepack_patterns:
            # Register to pass_number 1, so we can do dequant promotion in pass_number 0.
            _register_qconv_weight_prepack_pass(
                weight_prepack_pattern, pass_number=1, dtype=dtype
            )


def _register_qlinear_weight_prepack():
    # 6 Linear related patterns will be matched based on the dtype, input dimension size and input contiguous.
    # Then convert the pattern into a QLinear node with int8_fp32/bf16.
    # Case 1: int8-mixed-fp32, input dim size is 2
    # Case 2: int8-mixed-fp32, input dim size exceeds 2 and contiguous
    # Case 3: int8-mixed-bf16, input dim size is 2
    # Case 4: int8-mixed-bf16, input dim size exceeds 2 and contiguous

    #   + - - - - | - - - - - - | - - - - - +
    #   |    dq_per_tensor  dq_per_channel  |
    #   |         |              |          |
    #   |    OPT(to_bf16)    OPT(to_bf16)   |
    #   |         |              |          |
    #   |     OPT(reshape)   permute        |
    #   |            \        /             |
    #   |             addmm/mm              |
    #   |                |                  |
    #   |           OPT(reshape)            |

    # Case 5: int8-mixed-fp32, input dim size exceeds 2 and not contiguous
    # Case 6: int8-mixed-bf16, input dim size exceeds 2 and not contiguous

    #   + - - - - | - - - - - - | - - - - - +
    #   |    dq_per_tensor  dq_per_channel  |
    #   |         |              |          |
    #   |    OPT(to_bf16)    OPT(to_bf16)   |
    #   |         |              |          |
    #   |       expand       permute        |
    #   |          \             |          |
    #   |                    expand         |
    #   |                    /              |
    #   |               bmm                 |
    #   |                |                  |
    #   |            OPT(add)               |

    linear_weight_prepack_cases = itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    )

    # Step 1: register patterns from mm and addmm
    for dtype, input_dim_exceeds_two, is_tensor_overload in linear_weight_prepack_cases:
        weight_prepack_patterns = _generate_qlinear_weight_prepack_patterns(
            dtype,
            input_dim_exceeds_two,
            is_tensor_overload=is_tensor_overload,
        )
        for weight_prepack_pattern in weight_prepack_patterns:
            # Register to pass_number 1, so we can do dequant promotion in pass_number 0.
            _register_qlinear_weight_prepack_pass(
                weight_prepack_pattern,
                pass_number=1,
                dtype=dtype,
                input_dim_exceeds_two=input_dim_exceeds_two,
            )

    # Step 2: register patterns from bmm
    # Linear might be decomposed into bmm when input dim exceeds 2 and not contiguous
    # refer to:
    # https://github.com/pytorch/pytorch/blob/80c07df659362a95da7cd4f3ec367abfdace38c4/torch/_decomp/decompositions.py#L3965-L3968
    # in this case, we can convert it back to qlinear
    for dtype, with_bias, is_tensor_overload in itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    ):
        bmm_pattern = _generate_qlinear_weight_prepack_patterns(
            dtype=dtype,
            input_dim_exceeds_two=True,
            input_contiguous=False,
            with_bias=with_bias,
            is_tensor_overload=is_tensor_overload,
        )
        _register_qlinear_weight_prepack_pass(
            bmm_pattern,
            pass_number=1
            if with_bias
            else 2,  # if with_bias, there is an output add, so we should try to match it firstly
            dtype=dtype,
            input_dim_exceeds_two=True,
            input_contiguous=False,
        )


def _register_linear_dynamic_fp16_weight_prepack_pass(
    pattern,
    pass_number,
    input_dim_exceeds_two=False,
    input_contiguous=True,
    relu_fused=False,
):
    def _extra_check_fn(match: Match):
        return match.kwargs["dtype_fp16"] == torch.float16

    @register_freezing_graph_pattern(
        pattern,
        extra_check=_extra_check_fn,
        pass_number=pass_number,
    )
    def linear_dynamic_fp16_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        fp32 activation
          |
        mm/addmm <- t <- to_fp32 <- to_fp16 <- weight
          |
        (reshape) <- (relu)

        OR

        fp32 activation
          |
        expand
          |
         bmm <- expand <- t <- to_fp32 <- to_fp16 <- weight
          |
        (add) <- (relu)

        Insert weight prepack node and change the pattern to:
        fp32 activation
          |
        onednn.linear_dynamic_fp16 <- onednn.linear_prepack_fp16 <- weight
        (or onednn.linear_relu_dynamic_fp16)
        """
        # find params
        x = kwargs["x"]
        w = kwargs["w"]
        bias = kwargs["b"] if "b" in kwargs else None

        # find linear node
        nodes_to_find = [aten.addmm.default, aten.mm.default, aten.bmm.default]
        linear_nodes = []
        for node in nodes_to_find:
            linear_nodes.extend(filter_nodes(match.nodes, node))
        assert len(linear_nodes) == 1
        linear_node = linear_nodes[0]
        assert isinstance(linear_node, torch.fx.node.Node)
        input_index = 1 if linear_node.target is aten.addmm.default else 0
        weight_index = input_index + 1

        # find relu node
        relu_node = None
        if relu_fused:
            relu_node = match.output_node()
            assert isinstance(relu_node, torch.fx.node.Node)

        # find reshape node, expand node and add node
        (
            act_reshape_node,
            output_reshape_node,
            expand_x_node,
            expand_w_node,
            add_bias_node,
        ) = (None, None, None, None, None)
        t_node = None
        if input_dim_exceeds_two:
            if input_contiguous:
                act_reshape_node = linear_node.args[input_index]
                t_node = linear_node.args[weight_index]
                output_reshape_node = next(iter(linear_node.users))
                assert output_reshape_node.target is aten.reshape.default
            else:
                expand_x_node = linear_node.args[input_index]
                expand_w_node = linear_node.args[weight_index]
                assert isinstance(expand_w_node, torch.fx.node.Node)
                t_node = expand_w_node.args[0]
                if bias:
                    add_bias_node = next(iter(linear_node.users))
                    assert add_bias_node.target is aten.add.Tensor
        else:
            t_node = linear_node.args[weight_index]
        assert isinstance(t_node, torch.fx.node.Node)

        w_to_fp32_node = t_node.args[0]
        assert (
            isinstance(w_to_fp32_node, torch.fx.node.Node)
            and w_to_fp32_node.target
            is quantized_decomposed.convert_element_type.no_fuse
        )
        w_to_fp16_node = w_to_fp32_node.args[0]
        assert (
            isinstance(w_to_fp16_node, torch.fx.node.Node)
            and w_to_fp16_node.target
            is quantized_decomposed.convert_element_type.no_fuse
        )

        x_shape = x.meta.get("tensor_meta").shape
        if has_free_symbols(x_shape):
            # For dynamic shape case, we can't get activation shape ahead of runtime.
            x_shape = None
        graph = match.graph
        with graph.inserting_before(linear_node):
            # Insert weight prepack node and the qlinear node
            packed_weight_inputs = (
                w,
                x_shape,
            )
            packed_weight_op = torch.ops.onednn.linear_prepack_fp16
            prepack_weight_node = graph.call_function(
                packed_weight_op, args=packed_weight_inputs
            )

            # create new linear node and insert on graph
            new_args: tuple[Any, ...] = (
                x,
                prepack_weight_node,
                bias,
            )
            linear_op = (
                torch.ops.onednn.linear_relu_dynamic_fp16.default
                if relu_fused
                else torch.ops.onednn.linear_dynamic_fp16.default
            )
            new_linear_node = graph.call_function(linear_op, args=new_args)
            out_node = match.output_node()
            out_node.replace_all_uses_with(new_linear_node)

            # Erase the original nodes in the reverse order
            new_linear_node.meta.update(out_node.meta)
            if relu_node is not None:
                graph.erase_node(relu_node)
            if output_reshape_node is not None:
                graph.erase_node(output_reshape_node)
            if add_bias_node is not None:
                graph.erase_node(add_bias_node)
            graph.erase_node(linear_node)
            if act_reshape_node is not None:
                assert isinstance(act_reshape_node, torch.fx.node.Node)
                graph.erase_node(act_reshape_node)
            if expand_x_node is not None:
                assert isinstance(expand_x_node, torch.fx.node.Node)
                graph.erase_node(expand_x_node)
            if expand_w_node is not None:
                assert isinstance(expand_w_node, torch.fx.node.Node)
                graph.erase_node(expand_w_node)
            graph.erase_node(t_node)
            graph.erase_node(w_to_fp32_node)
            graph.erase_node(w_to_fp16_node)

            counters["inductor"]["qlinear_weight_prepack_matcher_count"] += 1
            counters["inductor"]["qlinear_weight_prepack_matcher_nodes"] += len(
                match.nodes
            )


def _register_linear_dynamic_fp16_weight_prepack():
    to_dtype_op = torch.ops.quantized_decomposed.convert_element_type.no_fuse
    weight_pattern = CallFunction(
        to_dtype_op,
        CallFunction(
            to_dtype_op,
            KeywordArg("w"),
            KeywordArg("dtype_fp16"),
        ),
        KeywordArg("dtype_fp32"),
    )
    cases = itertools.product(
        [False, True],  # input_dim_exceeds_two
        [True, False],  # input_contiguous
        [False, True],  # relu fused
    )
    for input_dim_exceeds_two, input_contiguous, relu_fused in cases:
        patterns = _generate_linear_dynamic_fp16_pattern(
            weight_pattern,
            input_dim_exceeds_two,
            input_contiguous,
            relu_fused,
        )
        for pattern in patterns:
            _register_linear_dynamic_fp16_weight_prepack_pass(
                pattern,
                pass_number=0 if relu_fused else 1,
                input_dim_exceeds_two=input_dim_exceeds_two,
                input_contiguous=input_contiguous,
                relu_fused=relu_fused,
            )


def _register_smooth_quant_int_mm_pattern():
    """
    The pattern is:
      (no bias) reshape -> _int_mm -> convert_element_type -> (expand ->) mul -> mul -> reshape
    or
      (with bias) pattern_no_bias -> add (-> reshape -> reshape)
    """

    # When torch.compile'ing with dynamic=True, the expand node and the two tailing reshape nodes exist
    # When torch.compile'ing with dynamic=False, they don't exist
    def get_pattern_no_bias(expand_a_scale: bool, reshape_a: bool = True):
        return CallFunction(
            aten.mul.Tensor,
            CallFunction(
                aten.mul.Tensor,
                CallFunction(
                    prims.convert_element_type.default,
                    CallFunction(
                        aten._int_mm.default,
                        CallFunction(
                            aten.reshape.default,
                            KeywordArg("a"),
                            KeywordArg("in_shape"),
                        )
                        if reshape_a
                        else KeywordArg("a"),
                        KeywordArg("b"),
                    ),
                    KeywordArg("dtype"),
                ),
                (
                    CallFunction(
                        aten.expand.default,
                        KeywordArg("x_scale"),
                        Arg(),
                    )
                    if expand_a_scale
                    else KeywordArg("x_scale")
                ),
            ),
            KeywordArg("w_scale"),
        )

    def _with_outer_reshape(pattern):
        return CallFunction(
            aten.reshape.default, pattern, KeywordArg("out_shape_no_bias")
        )

    # for torch.compile(dynamic=False)
    pattern_no_bias_1 = _with_outer_reshape(get_pattern_no_bias(expand_a_scale=False))
    pattern_with_bias_1 = CallFunction(
        aten.add.Tensor,
        pattern_no_bias_1,
        KeywordArg("bias"),
    )
    # for torch.compile(dynamic=True)
    pattern_no_bias_2 = _with_outer_reshape(get_pattern_no_bias(expand_a_scale=True))
    pattern_with_bias_2 = CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.add.Tensor,
                pattern_no_bias_2,
                KeywordArg("bias"),
            ),
            Arg(),
        ),
        KeywordArg("out_shape_with_bias"),
    )

    # The following patterns are for torchao int8_dynamic_activation_int8_weight linear,
    # when both activation and weights are symmetrically quantized.
    # In practice, though, they may also match smooth-quant pattern when a 2D input shape would be used.
    # Since add is not currently being used as a oneDNN post-op, but is unfused, we don't need these patterns with bias.
    # Ideally, we should add mul + add post-op support in ATen int8 oneDNN linear op.
    pattern1_with_no_outer_or_act_reshape = get_pattern_no_bias(
        expand_a_scale=False, reshape_a=False
    )
    pattern2_with_no_outer_or_act_reshape = get_pattern_no_bias(
        expand_a_scale=True, reshape_a=False
    )

    def _validate_pattern(match: Match):
        if len(match.nodes) not in [4, 5, 6, 7, 10]:
            return False
        # Make sure weight is a constant
        aten_int_mm_node = filter_nodes(match.nodes, aten._int_mm.default)[0]
        if not isinstance(aten_int_mm_node.args[1], torch.fx.node.Node):
            return False
        if aten_int_mm_node.args[1].op != "get_attr":
            return False

        if len(match.nodes) == 10:
            # Check the two tailing reshape nodes can be fused
            if match.nodes[9].args[1] != match.nodes[6].args[1]:
                return False
        if len(match.nodes) == 10 or (
            len(match.nodes) == 7 and match.nodes[6].target is aten.add.Tensor
        ):
            bias_idx = 7 if len(match.nodes) == 10 else 6
            # Check bias shape
            bias_node = match.nodes[bias_idx].args[1]
            if not isinstance(bias_node, torch.fx.node.Node):
                return False
            if len(bias_node.meta.get("tensor_meta").shape) != 1:  # type: ignore[union-attr]
                return False
        return True

    pattern_to_pass_number = {
        pattern_no_bias_2: 0,
        pattern_with_bias_2: 0,
        pattern_no_bias_1: 1,
        pattern_with_bias_1: 1,
        pattern1_with_no_outer_or_act_reshape: 2,
        pattern2_with_no_outer_or_act_reshape: 2,
    }
    for pattern, pass_number in pattern_to_pass_number.items():

        @register_freezing_graph_pattern(
            pattern,
            extra_check=_validate_pattern,
            pass_number=pass_number,
        )
        def _int_mm_weight_prepack(match: Match, *args, **kwargs):
            bias = kwargs.get("bias", None)
            x = kwargs["a"]
            weight = kwargs["b"]
            dtype = kwargs["dtype"]
            x_scale = kwargs["x_scale"]
            w_scale = kwargs["w_scale"]
            x_shape = x.meta.get("tensor_meta").shape
            if has_free_symbols(x_shape):
                # For dynamic shape case, we can't get activation shape ahead of runtime.
                x_shape = None

            out_node = match.output_node()
            with match.graph.inserting_before(out_node):
                transpose_node = match.graph.call_function(
                    aten.permute.default, args=(weight, [1, 0])
                )
                contig_node = match.graph.call_function(
                    aten.contiguous.default, args=(transpose_node,)
                )
                packed_weight_inputs = (
                    contig_node,
                    x_shape,
                )
                packed_weight_op = torch.ops.onednn.qlinear_prepack
                prepack_weight_node = match.graph.call_function(
                    packed_weight_op, args=packed_weight_inputs
                )

                dummy_zp = None
                w_scale = match.graph.call_function(
                    prims.convert_element_type.default, args=(w_scale, torch.float32)
                )

                x_scale_shape = x_scale.meta.get("tensor_meta").shape
                x_scale_is_scalar = False
                if not has_free_symbols(x_scale_shape):
                    prod = 1
                    for d in x_scale_shape:
                        prod *= d
                    x_scale_is_scalar = prod == 1

                new_args: tuple[Any, ...]
                if x_scale_is_scalar:
                    # in this case, we can call onednn.qlinear directly
                    new_args = (
                        x,
                        x_scale,
                        dummy_zp,  # x_zp
                        prepack_weight_node,
                        w_scale,
                        dummy_zp,  # w_zp
                        bias,
                        1.0,  # output_scale
                        0,  # output_zero_point
                        dtype,  # output_dtype
                        "none",  # post op name
                        [],  # post op args
                        "",  # post op algorithm
                    )
                    new_linear_node = match.graph.call_function(
                        torch.ops.onednn.qlinear_pointwise.tensor, args=new_args
                    )
                    out_node.replace_all_uses_with(new_linear_node)
                    new_linear_node.meta.update(out_node.meta)
                else:
                    # onednn.qlinear does not support per-channel quantization of x
                    # so in this case, we have to apply x scale and add bias ourselves after qlinear
                    in_shape = kwargs.get("in_shape", None)
                    if in_shape is None:
                        x_reshaped = x
                    else:
                        x_reshaped = match.graph.call_function(
                            aten.reshape.default, args=(x, in_shape)
                        )
                    new_args = (
                        x_reshaped,
                        1.0,  # x_scale
                        0,  # x_zp
                        prepack_weight_node,
                        w_scale,
                        dummy_zp,  # w_zp
                        None,  # bias
                        1.0,  # output_scale
                        0,  # output_zero_point
                        dtype,  # output_dtype
                        "none",  # post op name
                        [],  # post op args
                        "",  # post op algorithm
                    )
                    new_linear_node = match.graph.call_function(
                        torch.ops.onednn.qlinear_pointwise, args=new_args
                    )
                    # apply x scale
                    new_out_node = match.graph.call_function(
                        aten.mul.Tensor, args=(new_linear_node, x_scale)
                    )

                    # Add bias and reshape
                    has_outer_reshape = (
                        kwargs.get("out_shape_with_bias", None) is not None
                        or kwargs.get("out_shape_no_bias", None) is not None
                    )

                    if has_outer_reshape:
                        out_shape = kwargs.get(
                            "out_shape_with_bias", kwargs["out_shape_no_bias"]
                        )
                    if bias is not None:
                        new_out_node = match.graph.call_function(
                            aten.add.Tensor, args=(new_out_node, bias)
                        )
                        if has_outer_reshape:
                            new_out_node = match.graph.call_function(
                                aten.reshape.default,
                                args=(new_out_node, out_shape),  # type: ignore[possibly-undefined]
                            )
                    else:
                        if has_outer_reshape:
                            new_out_node = match.graph.call_function(
                                aten.reshape.default,
                                args=(new_out_node, out_shape),  # type: ignore[possibly-undefined]
                            )
                    out_node.replace_all_uses_with(new_out_node)
                    new_out_node.meta.update(out_node.meta)
                for node in reversed(match.nodes):
                    match.graph.erase_node(node)
                counters["inductor"]["qlinear_weight_prepack_matcher_count"] += 1
                counters["inductor"]["qlinear_weight_prepack_matcher_nodes"] += len(
                    match.nodes
                )


class PostOpAttr:
    def __init__(
        self,
        binary_op_name: str = "none",
        alpha=None,
        unary_op_name: str = "none",
        scalars_attr=None,
        algorithm_attr=None,
    ) -> None:
        self.binary_op_name = binary_op_name
        self.alpha = alpha if alpha else 1.0
        self.unary_op_name = unary_op_name
        self.scalars_attr = scalars_attr if scalars_attr else []
        self.algorithm_attr = algorithm_attr if algorithm_attr else ""


def _register_qconv_post_op_fusion_pass(
    pattern,
    pass_number,
    computation_op,
    post_op_attr,
):
    has_binary_post_op = post_op_attr.binary_op_name != "none"

    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_qconv_post_op_fusion_pattern(has_binary_post_op),
        pass_number=pass_number,
    )
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
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype in [torch.int8, torch.uint8, torch.float32, torch.bfloat16]
        # Output QParams
        o_inv_scale = (
            kwargs["o_inv_scale"]
            if (output_dtype == torch.uint8 or output_dtype == torch.int8)
            else 1.0
        )
        o_zero_point = (
            kwargs["o_zp"]
            if (output_dtype == torch.uint8 or output_dtype == torch.int8)
            else 0
        )
        assert (
            kwargs["postop_name"] == "none"
        )  # Expected no post op fused in weight prepack phase
        if post_op_attr.unary_op_name == "hardtanh":
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            post_op_attr.scalars_attr = [min_value, max_value]

        out_node = match.output_node()
        with match.graph.inserting_before(out_node):
            if not has_binary_post_op:
                computation_args: tuple[Any, ...] = (
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
                    output_dtype,
                    post_op_attr.unary_op_name,
                    post_op_attr.scalars_attr,
                    post_op_attr.algorithm_attr,
                )
            else:
                accum = (
                    kwargs["accum"]
                    if output_dtype in [torch.uint8, torch.int8]
                    else kwargs["accum_after_dequant"]
                )
                accum_scale = (
                    kwargs["accum_scale"]
                    if output_dtype in [torch.uint8, torch.int8]
                    else 1.0
                )
                accum_zp = (
                    kwargs["accum_zp"]
                    if output_dtype in [torch.uint8, torch.int8]
                    else 0
                )
                computation_args = (
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    accum,
                    b,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    accum_scale,
                    accum_zp,
                    post_op_attr.binary_op_name,
                    post_op_attr.alpha,
                    post_op_attr.unary_op_name,
                    post_op_attr.scalars_attr,
                    post_op_attr.algorithm_attr,
                )
            new_conv_node = match.graph.call_function(
                computation_op, args=computation_args
            )
            out_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(out_node.meta)
            for node in reversed(match.nodes):
                match.graph.erase_node(node)
        count_key = (
            "qconv2d_binary_matcher_count"
            if has_binary_post_op
            else "qconv_unary_matcher_count"
        )
        nodes_key = (
            "qconv2d_binary_matcher_nodes"
            if has_binary_post_op
            else "qconv_unary_matcher_nodes"
        )
        counters["inductor"][count_key] += 1
        counters["inductor"][nodes_key] += len(match.nodes)

    return qconv


def _register_qconv_unary_fusion():
    from .mkldnn_fusion import _hardswish_fusion, _hardtanh_fusion, _silu_fusion

    for original_pattern_output_dtype in [torch.float32, torch.bfloat16]:
        # Priority 1 to match: QConv2d Unary pattern with int8 output
        # If a pattern1 is a sub-set of pattern2, we should try to match pattern2 firstly.
        # For example: pattern1 is qconv_fp32 -> relu, pattern2 is qconv_fp32 -> relu -> quant
        is_bf16 = original_pattern_output_dtype == torch.bfloat16
        conv_unary_replace_patterns = {
            PostOpAttr(
                "none", None, "none", [], ""
            ): generate_pattern_with_output_quant(
                get_qconv_pt2e_pattern(1),
            ),
            PostOpAttr(
                "none", None, "relu", [], ""
            ): generate_pattern_with_output_quant(
                generate_pattern_with_unary(
                    get_qconv_pt2e_pattern(1), aten.relu.default
                ),
            ),
            PostOpAttr(
                "none", None, "hardtanh", [], ""
            ): generate_pattern_with_output_quant(
                _unary_fusion_pattern(
                    _hardtanh_fusion,
                    get_qconv_pt2e_pattern(1),
                    1,
                    is_bf16,
                ),
                with_dtype_convert=is_bf16,
            ),
            PostOpAttr(
                "none", None, "hardswish", [], ""
            ): generate_pattern_with_output_quant(
                _unary_fusion_pattern(
                    _hardswish_fusion,
                    get_qconv_pt2e_pattern(1 if is_bf16 else 2),
                    2,
                    is_bf16,
                ),
                with_dtype_convert=is_bf16,
            ),
            PostOpAttr(
                "none", None, "swish", [], ""
            ): generate_pattern_with_output_quant(
                _unary_fusion_pattern(
                    _silu_fusion,
                    get_qconv_pt2e_pattern(1 if is_bf16 else 2),
                    2,
                    is_bf16,
                ),
                with_dtype_convert=is_bf16,
            ),
        }

        for unary_attr, patterns in conv_unary_replace_patterns.items():
            # Register qconv2d pattern for ExternKernel Lowering
            _register_qconv_post_op_fusion_pass(
                patterns,
                3,  # pass_number
                torch.ops.onednn.qconv_pointwise.default,  # computation_op
                unary_attr,  # unary_attr
            )

        # Priority 2 to match: QConv2d Unary pattern with fp32/bfloat16 output
        conv_unary_replace_float_out_patterns = {
            PostOpAttr("none", None, "relu", [], ""): generate_pattern_with_unary(
                get_qconv_pt2e_pattern(1), aten.relu.default
            ),
            PostOpAttr(
                "none", None, "hardtanh", [], ""
            ): _may_generate_pattern_with_dtype_convert(
                _unary_fusion_pattern(
                    _hardtanh_fusion,
                    get_qconv_pt2e_pattern(1),
                    1,
                    is_bf16,
                ),
                Arg(),
                is_bf16,
            ),
            PostOpAttr(
                "none", None, "hardswish", [], ""
            ): _may_generate_pattern_with_dtype_convert(
                _unary_fusion_pattern(
                    _hardswish_fusion,
                    get_qconv_pt2e_pattern(1 if is_bf16 else 2),
                    2,
                    is_bf16,
                ),
                Arg(),
                is_bf16,
            ),
            PostOpAttr(
                "none", None, "swish", [], ""
            ): _may_generate_pattern_with_dtype_convert(
                _unary_fusion_pattern(
                    _silu_fusion,
                    get_qconv_pt2e_pattern(1 if is_bf16 else 2),
                    2,
                    is_bf16,
                ),
                Arg(),
                is_bf16,
            ),
        }

        for unary_attr, patterns in conv_unary_replace_float_out_patterns.items():
            # Register qconv2d pattern for ExternKernel Lowering
            _register_qconv_post_op_fusion_pass(
                patterns,
                4,  # pass_number
                torch.ops.onednn.qconv_pointwise.default,  # computation_op
                unary_attr,  # unary_attr
            )


def _register_qconv_binary_fusion():
    for int8_mixed_bf16_with_inplace_add in [False, True]:
        # Priority 1 to match: QConv2d Binary or Binary-Unary pattern with int8 output
        swap_binary_inputs_list = [False, True]
        binary_replace_patterns = {}
        for swap_inputs in swap_binary_inputs_list:
            binary_replace_patterns.update(
                {
                    PostOpAttr(
                        "sum", 1.0, "none", [], ""
                    ): generate_pattern_with_output_quant(
                        generate_pattern_with_binary(
                            aten.add.Tensor,
                            get_qconv_pt2e_pattern(1),
                            dequantize_accum_pattern,
                            int8_mixed_bf16_with_inplace_add,
                            swap_inputs=swap_inputs,
                        ),
                    ),
                    PostOpAttr(
                        "sum", 1.0, "relu", [], ""
                    ): generate_pattern_with_output_quant(
                        generate_pattern_with_unary(
                            generate_pattern_with_binary(
                                aten.add.Tensor,
                                get_qconv_pt2e_pattern(1),
                                dequantize_accum_pattern,
                                int8_mixed_bf16_with_inplace_add,
                                swap_inputs=swap_inputs,
                            ),
                            aten.relu.default,
                        ),
                    ),
                }
            )

        for binary_unary_attr, patterns in binary_replace_patterns.items():
            _register_qconv_post_op_fusion_pass(
                patterns,
                3,  # pass_number
                torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
                binary_unary_attr,  # binary_unary_attr
            )

        # Priority 2 to match: QConv2d Binary-Unary pattern with fp32/bfloat16 output
        binary_replace_float_out_patterns = {}
        for swap_inputs in swap_binary_inputs_list:
            binary_replace_float_out_patterns.update(
                {
                    PostOpAttr("sum", 1.0, "relu", [], ""): generate_pattern_with_unary(
                        generate_pattern_with_binary(
                            aten.add.Tensor,
                            get_qconv_pt2e_pattern(1),
                            KeywordArg("accum_after_dequant"),
                            int8_mixed_bf16_with_inplace_add,
                            swap_inputs=swap_inputs,
                        ),
                        aten.relu.default,
                    )
                }
            )

        for (
            binary_unary_attr,
            patterns,
        ) in binary_replace_float_out_patterns.items():
            if int8_mixed_bf16_with_inplace_add:
                _register_qconv_post_op_fusion_pass(
                    patterns,
                    3,  # pass_number
                    torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
                    binary_unary_attr,  # binary_unary_attr
                )
            else:
                _register_qconv_post_op_fusion_pass(
                    patterns,
                    4,  # pass_number
                    torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
                    binary_unary_attr,  # binary_unary_attr
                )

        # Priority 3: QConv2d Binary pattern with fp32/bfloat16 output
        binary_replace_float_out_patterns = {}
        for swap_inputs in swap_binary_inputs_list:
            binary_replace_float_out_patterns.update(
                {
                    PostOpAttr(
                        "sum", 1.0, "none", [], ""
                    ): generate_pattern_with_binary(
                        aten.add.Tensor,
                        get_qconv_pt2e_pattern(1),
                        KeywordArg("accum_after_dequant"),
                        int8_mixed_bf16_with_inplace_add,
                        swap_inputs=swap_inputs,
                    ),
                }
            )

        for (
            binary_unary_attr,
            patterns,
        ) in binary_replace_float_out_patterns.items():
            _register_qconv_post_op_fusion_pass(
                patterns,
                4 if int8_mixed_bf16_with_inplace_add else 5,  # pass_number
                torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
                binary_unary_attr,  # binary_unary_attr
            )


def _register_qlinear_post_op_fusion_pass(
    pattern,
    pass_number,
    computation_op,
    post_op_attr,
):
    has_binary_post_op = post_op_attr.binary_op_name != "none"

    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_qlinear_post_op_fusion_pattern(has_binary_post_op),
        pass_number=pass_number,
    )
    def qlinear_post_op_fusion(match: Match, *args, **kwargs):
        """
        Match the pattern:
        qlinear - post op
        """
        output_dtype = _get_pattern_output_dtype(match)
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

        # bias
        b = kwargs["b"] if "b" in kwargs else None

        # Output QParams
        o_inv_scale = (
            kwargs["o_inv_scale"]
            if (output_dtype in [torch.uint8, torch.int8])
            else 1.0
        )
        o_zero_point = (
            kwargs["o_zp"] if (output_dtype in [torch.uint8, torch.int8]) else 0
        )
        assert (
            kwargs["postop_name"] == "none"
        )  # Expected no post op fused in weight prepack phase

        out_node = match.output_node()
        with match.graph.inserting_before(out_node):
            if not has_binary_post_op:
                computation_args: tuple[Any, ...] = (
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    b,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    post_op_attr.unary_op_name,
                    post_op_attr.scalars_attr,
                    post_op_attr.algorithm_attr,
                )
            else:
                other = kwargs["other"] if "other" in kwargs else kwargs["accum"]
                x2_scale = 1.0
                x2_zp = 0
                computation_args = (
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    other,
                    b,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    x2_scale,
                    x2_zp,
                    post_op_attr.binary_op_name,
                    post_op_attr.alpha,
                    post_op_attr.unary_op_name,
                    post_op_attr.scalars_attr,
                    post_op_attr.algorithm_attr,
                )
            new_linear_node = match.graph.call_function(
                computation_op, args=computation_args
            )
            out_node.replace_all_uses_with(new_linear_node)
            new_linear_node.meta.update(out_node.meta)
            for node in reversed(match.nodes):
                match.graph.erase_node(node)
        count_key = (
            "qlinear_binary_matcher_count"
            if has_binary_post_op
            else "qlinear_unary_matcher_count"
        )
        nodes_key = (
            "qlinear_binary_matcher_nodes"
            if has_binary_post_op
            else "qlinear_unary_matcher_nodes"
        )
        counters["inductor"][count_key] += 1
        counters["inductor"][nodes_key] += len(match.nodes)


def _register_qlinear_unary_fusion():
    from .mkldnn_fusion import (
        _gelu_fusion_1 as _gelu_fusion_erf,
        _gelu_fusion_2 as _gelu_fusion_tanh,
    )

    for original_pattern_output_dtype in [torch.float32, torch.bfloat16]:
        is_bf16 = original_pattern_output_dtype == torch.bfloat16
        for x_scale_zp_are_tensors in (False, True):
            qlinear_pattern = get_qlinear_pt2e_pattern(x_scale_zp_are_tensors)
            computation_op = (
                torch.ops.onednn.qlinear_pointwise.tensor
                if x_scale_zp_are_tensors
                else torch.ops.onednn.qlinear_pointwise.default
            )
            # Priority 1 to match: QLinear Unary pattern with int8 output
            linear_unary_replace_patterns = {
                PostOpAttr(
                    "none", None, "none", [], ""
                ): generate_pattern_with_output_quant(
                    qlinear_pattern,
                ),
                PostOpAttr(
                    "none", None, "relu", [], ""
                ): generate_pattern_with_output_quant(
                    generate_pattern_with_unary(qlinear_pattern, aten.relu.default),
                ),
                PostOpAttr(
                    "none", None, "gelu", [], "none"
                ): generate_pattern_with_output_quant(
                    _unary_fusion_pattern(
                        _gelu_fusion_erf,
                        get_qlinear_pt2e_pattern(
                            x_scale_zp_are_tensors, 1 if is_bf16 else 2
                        ),
                        2,
                        is_bf16,
                    ),
                    with_dtype_convert=is_bf16,
                ),
                PostOpAttr(
                    "none", None, "gelu", [], "tanh"
                ): generate_pattern_with_output_quant(
                    _unary_fusion_pattern(
                        _gelu_fusion_tanh,
                        get_qlinear_pt2e_pattern(
                            x_scale_zp_are_tensors, 1 if is_bf16 else 4
                        ),
                        4,
                        is_bf16,
                    ),
                    with_dtype_convert=is_bf16,
                ),
            }

            for unary_attr, patterns in linear_unary_replace_patterns.items():
                _register_qlinear_post_op_fusion_pass(
                    patterns,
                    3,  # pass_number
                    computation_op,
                    unary_attr,  # unary_attr
                )

            # Priority 2 to match: QLinear Unary pattern with FP32/BF16 output
            linear_unary_replace_float_out_patterns = {
                PostOpAttr("none", None, "relu", [], ""): generate_pattern_with_unary(
                    qlinear_pattern, aten.relu.default
                ),
                PostOpAttr(
                    "none", None, "gelu", [], "none"
                ): _may_generate_pattern_with_dtype_convert(
                    _unary_fusion_pattern(
                        _gelu_fusion_erf,
                        get_qlinear_pt2e_pattern(
                            x_scale_zp_are_tensors, 1 if is_bf16 else 2
                        ),
                        2,
                        is_bf16,
                    ),
                    Arg(),
                    is_bf16,
                ),
                PostOpAttr(
                    "none", None, "gelu", [], "tanh"
                ): _may_generate_pattern_with_dtype_convert(
                    _unary_fusion_pattern(
                        _gelu_fusion_tanh,
                        get_qlinear_pt2e_pattern(
                            x_scale_zp_are_tensors, 1 if is_bf16 else 4
                        ),
                        4,
                        is_bf16,
                    ),
                    Arg(),
                    is_bf16,
                ),
            }

            for unary_attr, patterns in linear_unary_replace_float_out_patterns.items():
                _register_qlinear_post_op_fusion_pass(
                    patterns,
                    4,  # pass_number
                    computation_op,
                    unary_attr,  # unary_attr
                )


def _register_qlinear_binary_fusion():
    r"""
    Supported linear-binary(-unary) patterns

        linear(X)   extra input
               \   /
                Add
                 |
            Optional(relu)
                 |
                 Y

    1. int8-mixed-fp32
    +---+---------------+-----------+------------------------------+---------+
    | # | Add type      | Quant out | Pattern                      | Post op |
    +---+---------------+-----------+------------------------------+---------+
    | 1 | In-/out-place | Yes       | linear + fp32 -> (relu) -> q | add     |
    +---+---------------+-----------+------------------------------+---------+
    | 2 | In-/out-place | No        | linear + fp32 -> (relu)      | sum     |
    +---+---------------+-----------+------------------------------+---------+

    2. int8-mixed-bf16
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | # | X2 dtype | Add type      | Quant out | Pattern                                 | Post op |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 1 | BF16     | In-/out-place | Yes       | linear + bf16 -> (relu) -> q            | add     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 2 | BF16     | In-/out-place | No        | linear + bf16 -> (relu)                 | sum     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 3 | FP32     | Out-place     | Yes       | linear + fp32 -> (relu) -> q            | add     |
    |   |          | In-place right|           |                                         |         |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 4 | FP32     | Out-place     | No        | linear + fp32 -> (relu)                 | sum     |
    |   |          | In-place right|           |                                         |         |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 5 | FP32     | In-place left | Yes       | linear + fp32 -> to_bf16 -> (relu) -> q | add     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 6 | FP32     | In-place left | No        | linear + fp32 -> to_bf16 -> (relu)      | add     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+

    Note
    (1) The positions of linear and the extra input can be swapped.
    (2) we don't insert q-dq before the extra input of linear-add by recipe. But if q-dq is found at the
    extra input, we don't match that pattern because we cannot match all these patterns in 3 passes.
    """
    for x_scale_zp_are_tensors in (False, True):
        qlinear_binary_op = (
            torch.ops.onednn.qlinear_pointwise.binary_tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.binary
        )
        unary_postop_list = ["none", "relu"]
        unary_postop_dict = {
            "none": None,
            "relu": aten.relu.default,
        }
        convert_dtype_after_binary_list = [False, True]

        # Priority 1 to match: QLinear Binary or Binary-Unary pattern with int8 output
        # Covers case (1) of int8-mixed-fp32 and case (1)(3)(5) of int8-mixed-bf16,
        # totally 3 patterns (2 are identical)
        swap_binary_inputs_list = [False, True]
        int8_mixed_bf16_list = [False, True]
        combinations = itertools.product(
            unary_postop_list,
            int8_mixed_bf16_list,
            swap_binary_inputs_list,
            convert_dtype_after_binary_list,
        )
        qlinear_binary_replace_patterns = {}
        for unary_op, int8_mixed_bf16, swap_inputs, cvt_dtype_binary in combinations:
            if not int8_mixed_bf16 and cvt_dtype_binary:
                # No convert node after binary node if dtypes are all fp32
                continue
            qlinear_binary_replace_patterns.update(
                {
                    PostOpAttr(
                        "add", 1.0, unary_op, [], ""
                    ): generate_pattern_with_output_quant(
                        generate_pattern_with_unary(
                            generate_pattern_with_binary(
                                aten.add.Tensor,
                                get_qlinear_pt2e_pattern(x_scale_zp_are_tensors),
                                KeywordArg("other"),
                                # If fp32 extra input is inplace added to bf16 linear output,
                                # a to_bf16 node is inserted after binary
                                dtype_convert=cvt_dtype_binary,
                                swap_inputs=swap_inputs,
                            ),
                            unary_postop_dict[unary_op],
                        ),
                    )
                }
            )
        for binary_unary_attr, patterns in qlinear_binary_replace_patterns.items():
            _register_qlinear_post_op_fusion_pass(
                patterns,
                3,  # pass_number
                qlinear_binary_op,  # computation_op
                binary_unary_attr,
            )

        # Priority 2.1 to match: QLinear Binary-Unary pattern with fp32/bfloat16 output
        # Covers case (2) of int8-mixed-fp32 and case (2)(4) of int8-mixed-bf16,
        # totally 2 patterns (2 are identical)
        binary_replace_float_out_patterns = {}
        for swap_binary_inputs in swap_binary_inputs_list:
            binary_replace_float_out_patterns.update(
                {
                    PostOpAttr("sum", 1.0, "relu", [], ""): generate_pattern_with_unary(
                        generate_pattern_with_binary(
                            aten.add.Tensor,
                            get_qlinear_pt2e_pattern(x_scale_zp_are_tensors),
                            KeywordArg("accum"),
                            dtype_convert=False,
                            swap_inputs=swap_binary_inputs,
                        ),
                        aten.relu.default,
                    ),
                }
            )
        for (
            binary_unary_attr,
            patterns,
        ) in binary_replace_float_out_patterns.items():
            _register_qlinear_post_op_fusion_pass(
                patterns,
                4,  # pass_number
                qlinear_binary_op,  # computation_op
                binary_unary_attr,
            )
        # Priority 2.2 to match: QLinear Binary-Unary pattern with fp32/bfloat16 output
        # Covers case (6) of int8-mixed-bf16
        binary_replace_float_out_patterns = {}
        for swap_binary_inputs in swap_binary_inputs_list:
            binary_replace_float_out_patterns.update(
                {
                    PostOpAttr("add", 1.0, "relu", [], ""): generate_pattern_with_unary(
                        generate_pattern_with_binary(
                            aten.add.Tensor,
                            get_qlinear_pt2e_pattern(x_scale_zp_are_tensors),
                            KeywordArg("other"),
                            dtype_convert=True,
                            swap_inputs=swap_binary_inputs,
                        ),
                        aten.relu.default,
                    ),
                }
            )
        for (
            binary_unary_attr,
            patterns,
        ) in binary_replace_float_out_patterns.items():
            _register_qlinear_post_op_fusion_pass(
                patterns,
                4,  # pass_number
                qlinear_binary_op,  # computation_op
                binary_unary_attr,
            )

        # Priority 3.1: QLinear Binary pattern with fp32/bfloat16 output
        # Covers case (2) of int8-mixed-fp32 and case (2)(4) of int8-mixed-bf16,
        # totally 2 patterns (2 are identical)
        binary_replace_float_out_patterns = {}
        for swap_binary_inputs in swap_binary_inputs_list:
            binary_replace_float_out_patterns.update(
                {
                    PostOpAttr(
                        "sum", 1.0, "none", [], ""
                    ): generate_pattern_with_binary(
                        aten.add.Tensor,
                        get_qlinear_pt2e_pattern(x_scale_zp_are_tensors),
                        KeywordArg("accum"),
                        dtype_convert=False,
                        swap_inputs=swap_binary_inputs,
                    ),
                }
            )
        for (
            binary_unary_attr,
            patterns,
        ) in binary_replace_float_out_patterns.items():
            _register_qlinear_post_op_fusion_pass(
                patterns,
                5,  # pass_number
                qlinear_binary_op,  # computation_op
                binary_unary_attr,
            )
        # Priority 3.2: QLinear Binary pattern with fp32/bfloat16 output
        # Covers (6) of int8-mixed-bf16
        binary_replace_float_out_patterns = {}
        for swap_binary_inputs in swap_binary_inputs_list:
            binary_replace_float_out_patterns.update(
                {
                    PostOpAttr(
                        "add", 1.0, "none", [], ""
                    ): generate_pattern_with_binary(
                        aten.add.Tensor,
                        get_qlinear_pt2e_pattern(x_scale_zp_are_tensors),
                        KeywordArg("other"),
                        dtype_convert=True,
                        swap_inputs=swap_binary_inputs,
                    ),
                }
            )
        for (
            binary_unary_attr,
            patterns,
        ) in binary_replace_float_out_patterns.items():
            _register_qlinear_post_op_fusion_pass(
                patterns,
                5,  # pass_number
                qlinear_binary_op,  # computation_op
                binary_unary_attr,
            )


@functools.lru_cache(None)
def _register_quantization_weight_pack_pass():
    # Step 1: Dequant promotion for int8-mixed-fp32/bf16
    _register_dequant_promotion()

    # Step 2: QConv weight prepack
    _register_qconv_weight_prepack()

    # Step 3: QLinear weight prepack
    _register_qlinear_weight_prepack()
    _register_linear_dynamic_fp16_weight_prepack()

    # Step 4: weight prepack for SmoothQuant from Torchao
    _register_smooth_quant_int_mm_pattern()

    # Step 5: QLinear post op Fusion
    if not torch.ops.mkldnn._is_mkldnn_acl_supported():
        # skip fusion on ARM
        _register_qconv_unary_fusion()
        _register_qconv_binary_fusion()
        _register_qlinear_unary_fusion()
        _register_qlinear_binary_fusion()


def quant_lift_up(graph_module: torch.fx.GraphModule):
    """
    Lift up the quant node before view like nodes. It can benefit performance
    of Attention like block. For example, we have the pattern as:

             DQ
    DQ       LINEAR
    LINEAR   VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    Q        Q
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    We want to lift up the the quant nodes from matmul before view like nodes
    as the output of Linear node.

             DQ
    DQ       LINEAR
    LINEAR   Q
    Q        VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    It produces a DQ->LINEAR->Q pattern which can be fused by backend.
    """

    def is_view_op(node):
        return node.op == "call_function" and node.target in _VIEW_OPS

    for node in graph_module.graph.nodes:
        # <TODO> Leslie: Here we verify that the quant node has exactly
        # one input FX node, with constant scalar value for scale and zero point.
        # For the case input of quant node has more than one input FX nodes,
        # extend the implementation to lift up all the connected nodes
        # before the view nodes to keep the topological order.
        if (
            node.op == "call_function"
            and node.target in _PER_TENSOR_QUANTIZE_OPS
            and len(node.all_input_nodes) == 1
            and is_view_op(node.all_input_nodes[0])
        ):
            quant_node = node
            input_node_of_quant = quant_node.args[0]

            # Check the nodes along lift up path has only 1 user node
            # Propagate view like node to find where to insert the new quant node
            could_lift_up = True
            current_node = quant_node
            input_node = current_node.args[0]
            while is_view_op(input_node):
                if len(input_node.users) != 1:
                    could_lift_up = False
                    break
                current_node = input_node
                input_node = current_node.args[0]

            # Further check the input node of the first view node has only 1 user node
            if could_lift_up and len(input_node.users) == 1:
                # Replace dequant's input from quant to quant's input
                quant_node.replace_all_uses_with(input_node_of_quant)
                # Insert the new quant node
                with graph_module.graph.inserting_before(current_node):
                    new_quant_node = graph_module.graph.node_copy(quant_node)
                    input_node.replace_all_uses_with(new_quant_node)

                    # Update inputs of new_quant_node
                    def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
                        if n == input_node_of_quant:
                            return input_node
                        else:
                            return n

                    new_args = map_arg(new_quant_node.args, maybe_replace_node)
                    new_kwargs = map_arg(new_quant_node.kwargs, maybe_replace_node)
                    new_quant_node.args = new_args  # type: ignore[assignment]
                    new_quant_node.kwargs = new_kwargs  # type: ignore[assignment]
                    graph_module.graph.erase_node(quant_node)

    graph_module.graph.lint()
    graph_module.recompile()
