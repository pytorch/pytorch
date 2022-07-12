import sys
import warnings
from typing import Sequence

import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C

# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
from torch.onnx import _patch_torch  # noqa: F401
from torch.onnx import symbolic_helper
from torch.onnx import symbolic_opset9 as opset9
from torch.onnx._globals import GLOBALS

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


def div(g, self, other, *args):
    if len(args) == 0:
        return opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@symbolic_helper.parse_args("v", "v", "s")
def _div_rounding_mode(g, self, other, rounding_mode):
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return opset9._div_rounding_mode(g, self, other, rounding_mode)


def _floor_divide(g, self, other):
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = opset9.true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does trunction rounding
        div = g.op("Div", self, other)
        # Division is negative if: self < 0 != other < 0
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op("Xor", g.op("Less", self, zero), g.op("Less", other, zero))

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        mod = g.op("Mod", self, other, fmod_i=0)
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op("Sub", div, one)
        return g.op("Where", fixup_mask, fixup, div)


@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g, self, dim, decending, out=None):
    return symbolic_helper._sort_helper(g, self, dim, decending=decending, out=out)


@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g, self, k, dim, largest, sorted, out=None):
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


def _max_pool(name, tuple_fn, ndims, return_indices):
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if not stride:
            stride = kernel_size
        kwargs = {
            "kernel_shape_i": tuple_fn(kernel_size),
            "pads_i": tuple_fn(padding) * 2,
            "strides_i": tuple_fn(stride),
            "ceil_mode_i": ceil_mode,
        }
        if set(tuple_fn(dilation)) != {1}:
            kwargs["dilations_i"] = tuple_fn(dilation)
        # easy but hacky way to get flattened indices values
        # to be used to convert the indices values to non-flattened.
        # In ONNX the indices are computed as a flatten 1-D tensor,
        # so the values in indices are in [0, N x C x D1 x ... x Dn).
        # To convert the indices to the same format used by Pytorch,
        # we first execute a maxpool with a kernel and stride of 1 on the same input.
        # This will result in a tensor of indices in which each index will have it's own value.
        # Using this tensor as a reference, we extract the first index of each axis and subtract
        # it from each index of this axis in the indices to convert.
        # This step will result in a tensor were each dimension has values of indices within
        # the dimension it is in.
        # For more information :
        # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
        if return_indices:
            r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
            _, flattened_indices = g.op(
                "MaxPool",
                input,
                outputs=2,
                kernel_shape_i=[1 for _ in range(ndims)],
                strides_i=[1 for _ in range(ndims)],
            )
            # convert indices to have non-flattened indices values
            s = symbolic_helper._slice_helper(
                g,
                flattened_indices,
                axes=[2 + i for i in range(ndims)],
                starts=tuple_fn(0),
                ends=tuple_fn(1),
            )
            indices = opset9.sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d = _max_pool(
    "max_pool1d", torch.nn.modules.utils._single, 1, return_indices=False
)
max_pool2d = _max_pool(
    "max_pool2d", torch.nn.modules.utils._pair, 2, return_indices=False
)
max_pool3d = _max_pool(
    "max_pool3d", torch.nn.modules.utils._triple, 3, return_indices=False
)
max_pool1d_with_indices = _max_pool(
    "max_pool1d_with_indices", torch.nn.modules.utils._single, 1, return_indices=True
)
max_pool2d_with_indices = _max_pool(
    "max_pool2d_with_indices", torch.nn.modules.utils._pair, 2, return_indices=True
)
max_pool3d_with_indices = _max_pool(
    "max_pool3d_with_indices", torch.nn.modules.utils._triple, 3, return_indices=True
)


def _avg_pool(name, tuple_fn):
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Sequence[int],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        if not stride:
            stride = kernel_size
        padding = symbolic_helper._avgpool_helper(
            tuple_fn, padding, kernel_size, stride, divisor_override, name
        )
        if count_include_pad:
            input = opset9.op_with_optional_float_cast(
                g,
                "Pad",
                input,
                pads_i=((0,) * 2 + padding) * 2,
                mode_s="constant",
                value_f=0.0,
                opset_before=11,
            )
            padding = (0,) * len(padding)
        output = g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=padding * 2,
            ceil_mode_i=ceil_mode,
        )
        return output

    return symbolic_fn


avg_pool1d = _avg_pool("avg_pool1d", torch.nn.modules.utils._single)
avg_pool2d = _avg_pool("avg_pool2d", torch.nn.modules.utils._pair)
avg_pool3d = _avg_pool("avg_pool3d", torch.nn.modules.utils._triple)


def _interpolate(name, dim, interpolate_mode):
    @symbolic_helper.quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True")
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Resize", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


upsample_nearest1d = _interpolate("upsample_nearest1d", 3, "nearest")
upsample_nearest2d = _interpolate("upsample_nearest2d", 4, "nearest")
upsample_nearest3d = _interpolate("upsample_nearest3d", 5, "nearest")
upsample_linear1d = _interpolate("upsample_linear1d", 3, "linear")
upsample_bilinear2d = _interpolate("upsample_bilinear2d", 4, "linear")
upsample_trilinear3d = _interpolate("upsample_trilinear3d", 5, "linear")


def __interpolate(
    g, input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias
):
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Resize", input, scales, mode_s=mode)


def _slice(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    if dynamic_slice:
        starts = symbolic_helper._unsqueeze_helper(g, starts, [0])
        ends = symbolic_helper._unsqueeze_helper(g, ends, [0])
        if isinstance(axes, int):
            axes = g.op("Constant", value_t=torch.tensor(axes))
        axes = symbolic_helper._unsqueeze_helper(g, axes, [0])
    else:
        assert len(starts) == len(ends)
        assert len(starts) == len(axes)
        assert steps is None or len(starts) == len(steps)
        if (
            len(starts) == 1
            and starts[0] == 0
            and ends[0] == 9223372036854775807
            and (steps is None or (len(steps) == 1 and steps[0] == 1))
        ):
            return input
        axes = g.op("Constant", value_t=torch.tensor(axes))
        starts = g.op("Constant", value_t=torch.tensor(starts))
        ends = g.op("Constant", value_t=torch.tensor(ends))
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    steps = g.op("Constant", value_t=torch.tensor(steps))
    return g.op("Slice", input, starts, ends, axes, steps)


def slice(g, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor
        dim, start, end, step = args
    elif len(args) == 3:
        # aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]
        start, end, step = args
        dim = 0
    else:
        raise NotImplementedError("Unknown aten::slice signature")
    is_start_none = (
        start.node().kind() == "prim::Constant" and start.type().kind() == "NoneType"
    )
    is_end_none = (
        end.node().kind() == "prim::Constant" and end.type().kind() == "NoneType"
    )
    is_start_onnx_const = start.node().kind() == "onnx::Constant"
    is_end_onnx_const = end.node().kind() == "onnx::Constant"
    step = symbolic_helper._parse_arg(step, "i")
    if (
        (not is_start_none and not is_start_onnx_const)
        or (not isinstance(end, int) and not is_end_none and not is_end_onnx_const)
        or (not isinstance(dim, int) and dim.node().kind() != "onnx::Constant")
    ):
        dynamic_slice = True
        if is_start_none:
            start = g.op("Constant", value_t=torch.tensor(0))
        if is_end_none:
            end = g.op("Constant", value_t=torch.tensor(9223372036854775807))
    else:
        start = [0 if is_start_none else symbolic_helper._parse_arg(start, "i")]
        end = [
            9223372036854775807 if is_end_none else symbolic_helper._parse_arg(end, "i")
        ]
        dim = [symbolic_helper._parse_arg(dim, "i")]
        dynamic_slice = False
    return symbolic_helper._slice_helper(
        g,
        self,
        axes=dim,
        starts=start,
        ends=end,
        steps=[step],
        dynamic_slice=dynamic_slice,
    )


@symbolic_helper.parse_args("v", "is")
def flip(g, input, dims):
    return symbolic_helper._slice_helper(
        g,
        input,
        axes=dims,
        starts=[-1] * len(dims),
        ends=[-9223372036854775807] * len(dims),
        steps=[-1] * len(dims),
    )


def fmod(g, input, other):
    return g.op("Mod", input, other, fmod_i=1)


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    if scale_grad_by_freq and GLOBALS.training_mode:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    warnings.warn(
        "Export of embedding_bag with dynamic input/offsets shape is not supported in opset 10. "
        "Please use opset 11 or higher to export model for dynamic input shape.'"
    )
    offsets_dim_0 = symbolic_helper._get_tensor_dim_size(offsets, 0)
    if offsets_dim_0 is not None:
        if include_last_offset:
            offset_len = offsets_dim_0 - 1
            offsets_extended = offsets
        else:
            offset_len = offsets_dim_0
            offsets_extended = [
                offsets,
                g.op("Constant", value_t=torch.tensor([sys.maxsize])),
            ]
            offsets_extended = g.op("Concat", *offsets_extended, axis_i=0)
        list_ = []
        for i in range(offset_len):
            start_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(g, offsets_extended, torch.tensor(0), torch.tensor(i)),
                [0],
            )
            end_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(
                    g, offsets_extended, torch.tensor(0), torch.tensor(i + 1)
                ),
                [0],
            )
            axes_ = g.op("Constant", value_t=torch.tensor([0]))
            indices_row = g.op("Slice", indices, start_, end_, axes_)

            embeddings = g.op("Gather", embedding_matrix, indices_row)
            if not symbolic_helper._is_none(per_sample_weights):
                per_sample_weights_row = g.op(
                    "Slice", per_sample_weights, start_, end_, axes_
                )
                per_sample_weights_row = symbolic_helper._unsqueeze_helper(
                    g, per_sample_weights_row, [1]
                )
                embeddings = g.op("Mul", embeddings, per_sample_weights_row)
            if mode == 0:
                embeddings = symbolic_helper._reducesum_helper(
                    g, embeddings, axes_i=[0], keepdims_i=0
                )
            elif mode == 1:
                embeddings = g.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
            else:
                embeddings = g.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

            embeddings = symbolic_helper._unsqueeze_helper(g, embeddings, [0])
            list_.append(embeddings)

        output = g.op("Concat", *list_, axis_i=0)
        # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
        # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
        return output, None, None, None
    else:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with unknown shape of offsets for opset 10 is not supported. "
            "please use opset 11 or higher."
        )


@symbolic_helper.parse_args("v", "v", "v", "i", "i")
def fake_quantize_per_tensor_affine(
    g, inputs, scale, zero_point, quant_min=-128, quant_max=127
):
    # NOTE: (0, 127) is a special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) == (0, 127):
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Quantize range (0, 127) not supported, requires opset 13 Clip",
        )
    if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
        raise RuntimeError(
            "For (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). "
            "Got ({}, {})".format(quant_min, quant_max)
        )
    scale = symbolic_helper._maybe_get_scalar(scale)
    if scale is None:
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Non-constant scale not supported",
        )
    scale = scale.float().data  # Avoid exporter generating double type
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point),
        scale,
        zero_point,
    )


def isinf(g, input):
    return g.op("IsInf", opset9._cast_Double(g, input, False))  # type: ignore[attr-defined]


def isfinite(g, input):
    from torch.onnx.symbolic_opset9 import __not_, __or_

    inf_node = isinf(g, input)
    nan_node = opset9.isnan(g, input)
    return __not_(g, __or_(g, inf_node, nan_node))


def quantize_per_tensor(g, input, scale, zero_point, dtype):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    zero_point = g.op(
        "Cast", zero_point, to_i=symbolic_helper.scalar_type_to_onnx[dtype]
    )
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return symbolic_helper.quantize_helper(g, input, scale, zero_point)


def dequantize(g, input):
    return symbolic_helper.dequantize_helper(g, input)[0]


@symbolic_helper.parse_args("v", "f", "f", "f")
def nan_to_num(g, input, nan, posinf, neginf):
    # Cannot create a int type tensor with inf/nan values, so we simply
    # return the original tensor
    if not symbolic_helper._is_fp(input):
        return input
    input_dtype = symbolic_helper.pytorch_name_to_type[input.type().scalarType()]
    if nan is None:
        nan = 0.0
    nan_cond = opset9.isnan(g, input)
    nan_result = g.op(
        "Where",
        nan_cond,
        g.op("Constant", value_t=torch.tensor([nan], dtype=input_dtype)),
        input,
    )

    # For None values of posinf, neginf we use the greatest/lowest finite
    # value representable by input’s dtype.
    finfo = torch.finfo(input_dtype)
    if posinf is None:
        posinf = finfo.max
    posinf_cond = opset9.logical_and(
        g,
        isinf(g, nan_result),
        opset9.gt(g, nan_result, g.op("Constant", value_t=torch.LongTensor([0]))),
    )
    nan_posinf_result = g.op(
        "Where",
        posinf_cond,
        g.op("Constant", value_t=torch.tensor([posinf], dtype=input_dtype)),
        nan_result,
    )

    if neginf is None:
        neginf = finfo.min
    neginf_cond = opset9.logical_and(
        g,
        isinf(g, nan_posinf_result),
        opset9.lt(
            g, nan_posinf_result, g.op("Constant", value_t=torch.LongTensor([0]))
        ),
    )
    return g.op(
        "Where",
        neginf_cond,
        g.op("Constant", value_t=torch.tensor([neginf], dtype=input_dtype)),
        nan_posinf_result,
    )


# https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
class Quantized:
    """
    https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export

    Support starts from opset 10 because `DequantizeLinear` and `QuantizeLinear` were introduced in opset version 10.
    """

    domain = "quantized"

    @staticmethod
    def linear(g, q_input, q_weight, bias, op_scale, op_zero_point):
        input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
        weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
        q_bias = symbolic_helper.requantize_bias_helper(
            g, bias, input_scale, weight_scale
        )
        bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

        output = opset9.linear(g, input, weight, bias)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    def add(g, x, y, op_scale, op_zero_point):
        x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
        y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

        output = opset9.add(g, x, y)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    def add_relu(g, x, y, op_scale, op_zero_point):
        x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
        y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

        output = opset9.add(g, x, y)
        output = opset9.relu(g, output)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    def mul(g, x, y, op_scale, op_zero_point):
        x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
        y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

        output = opset9.mul(g, x, y)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    def hardswish(g, x, op_scale, op_zero_point):
        x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

        output = opset9.hardswish(g, x)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    def conv2d_relu(
        g,
        q_input,
        q_weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        op_scale,
        op_zero_point,
    ):
        input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
        weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
        q_bias = symbolic_helper.requantize_bias_helper(
            g, bias, input_scale, weight_scale
        )
        bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

        output = opset9.conv2d(
            g, input, weight, bias, stride, padding, dilation, groups
        )
        output = opset9.relu(g, output)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    def conv2d(
        g,
        q_input,
        q_weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        op_scale,
        op_zero_point,
    ):
        input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
        weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
        q_bias = symbolic_helper.requantize_bias_helper(
            g, bias, input_scale, weight_scale
        )
        bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

        output = opset9.conv2d(
            g, input, weight, bias, stride, padding, dilation, groups
        )

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

    @staticmethod
    @symbolic_helper.parse_args("v", "i", "v", "v")
    def cat(
        g,
        q_inputs: _C.Value,
        dim: int,
        op_scale: _C.Value,
        op_zero_point: _C.Value,
    ) -> _C.Value:
        unpacked_inputs = symbolic_helper._unpack_list(q_inputs)
        dequantized = [
            symbolic_helper.dequantize_helper(g, input)[0] for input in unpacked_inputs
        ]
        concatenated = g.op("Concat", *dequantized, axis_i=dim)
        return symbolic_helper.quantize_helper(g, concatenated, op_scale, op_zero_point)
