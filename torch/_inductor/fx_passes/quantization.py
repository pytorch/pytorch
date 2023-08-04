import torch
from ..ir import QConv
from ..pattern_matcher import Arg, CallFunction, KeywordArg, Match
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


def register_quantization_lowerings():
    _register_quantized_conv_lowering(quantize_conv_output_pattern)
