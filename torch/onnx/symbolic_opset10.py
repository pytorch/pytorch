# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
from torch.onnx import (
    _constants,
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset9 as opset9,
)
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


__all__ = [
    "dequantize",
    "div",
    "embedding_bag",
    "fake_quantize_per_tensor_affine",
    "flip",
    "fmod",
    "isfinite",
    "isinf",
    "nan_to_num",
    "quantize_per_tensor",
    "quantized_add_relu",
    "quantized_add",
    "quantized_cat",
    "quantized_conv1d_relu",
    "quantized_conv2d_relu",
    "quantized_conv3d_relu",
    "quantized_conv1d",
    "quantized_conv2d",
    "quantized_conv3d",
    "quantized_conv_transpose1d",
    "quantized_conv_transpose2d",
    "quantized_conv_transpose3d",
    "quantized_group_norm",
    "quantized_hardswish",
    "quantized_instance_norm",
    "quantized_layer_norm",
    "quantized_leaky_relu",
    "quantized_linear",
    "quantized_linear_relu",
    "quantized_mul",
    "quantized_sigmoid",
    "slice",
    "sort",
    "topk",
]


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=10)


@_onnx_symbolic("aten::div")
@_beartype.beartype
def div(g: jit_utils.GraphContext, self, other, *args):
    if len(args) == 0:
        return opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@symbolic_helper.parse_args("v", "v", "s")
@_beartype.beartype
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return opset9._div_rounding_mode(g, self, other, rounding_mode)


@_onnx_symbolic("aten::_floor_divide")
@_beartype.beartype
def _floor_divide(g: jit_utils.GraphContext, self, other):
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


@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def sort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    return symbolic_helper._sort_helper(g, self, dim, decending=decending, out=out)


@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
@_beartype.beartype
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


def _aten_max_pool_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> _C.Value:
    self_rank = g.op("Size", g.op("Shape", self))
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    pool_result, _ = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )

    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze",
            pool_result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    return pool_result


# For MaxPool
def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(dilation, int):
        dilation = [dilation] * expand_size

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2  # type: ignore[operator, assignment]
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2  # type: ignore[operator, assignment]
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding  # type: ignore[assignment]

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    return (kernel_shape, strides, pads, dilation)


def _aten_max_pool_with_indices_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
) -> Tuple[_C.Value, Sequence[int]]:
    self_rank = g.op("Size", g.op("Shape", self))
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    pool_result, indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )
    _, flatten_indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        dilations_i=dilations,
        kernel_shape_i=n_dims_one,
        strides_i=n_dims_one,
    )

    ends = g.op("Constant", value_t=torch.tensor(n_dims_one))
    starts = g.op("Constant", value_t=torch.tensor(n_dims_zero))
    axes = g.op("Constant", value_t=torch.tensor(n_dims_axes))

    delta = g.op("Slice", flatten_indices, starts, ends, axes)
    indices = g.op("Sub", indices, delta)

    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze", pool_result, value_t=torch.tensor([0], dtype=torch.int64)
        )
        indices = g.op("Squeeze", indices, value_t=torch.tensor([0], dtype=torch.int64))

    return (pool_result, indices)


@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[symbolic_helper._apply_params("max_pool1d", 1, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[symbolic_helper._apply_params("max_pool2d", 2, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[symbolic_helper._apply_params("max_pool3d", 3, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool1d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d_with_indices",
            1,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d_with_indices",
            2,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d_with_indices",
            3,
            return_indices=True,
        )
    ],
)
@_beartype.beartype
def _max_pool(name: str, expand_size: int, return_indices: bool):
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(
        g: jit_utils.GraphContext,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Union[int, Sequence[int]],
        dilation: Sequence[int],
        ceil_mode: bool,
    ):
        kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
            expand_size, kernel_size, stride, padding, dilation
        )

        if return_indices:
            return _aten_max_pool_with_indices_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
                ([1] * expand_size),
                ([0] * expand_size),
                ([2 + i for i in range(expand_size)]),
            )
        else:
            return _aten_max_pool_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
            )

    return symbolic_fn


# For AvgPool
def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        pads = padding * expand_size  # type: ignore[operator, assignment]
    else:
        pads = padding * 2  # type: ignore[operator, assignment]

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    return (kernel_shape, strides, pads)


@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[symbolic_helper._apply_params("avg_pool1d", 1)],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[symbolic_helper._apply_params("avg_pool2d", 2)],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[symbolic_helper._apply_params("avg_pool3d", 3)],
)
@_beartype.beartype
def _avg_pool(name, expand_size):
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    @_beartype.beartype
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Union[int, Sequence[int]],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
            expand_size, kernel_size, stride, padding
        )

        result = g.op(
            "AveragePool",
            input,
            ceil_mode_i=ceil_mode,
            count_include_pad_i=count_include_pad,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
        )

        return result

    return symbolic_fn


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
@_beartype.beartype
def _interpolate(name, dim, interpolate_mode):
    @symbolic_helper.quantized_args(True, False, False)
    @_beartype.beartype
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Resize", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


@_onnx_symbolic("aten::__interpolate")
@_beartype.beartype
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Resize", input, scales, mode_s=mode)


@_beartype.beartype
def _slice(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    axes: Union[List, torch.Tensor, torch._C.Value],
    starts: Union[List, torch.Tensor, torch._C.Value],
    ends: Union[List, torch.Tensor, torch._C.Value],
    steps: Optional[Union[List, torch.Tensor, torch._C.Value]] = None,
):
    def is_none_value(value):
        if value is None:
            return True
        return (
            isinstance(value, torch._C.Value)
            and value.node().kind() == "prim::Constant"
            and isinstance(value.type(), _C.NoneType)
        )

    def to_slice_input(list_or_value, default_value=None):
        # Convert input param into a 1D torch.Value.
        if is_none_value(list_or_value) and default_value is not None:
            list_or_value = [default_value]

        if isinstance(list_or_value, (list, torch.Tensor)):
            return g.op("Constant", value_t=torch.tensor(list_or_value))

        rank = symbolic_helper._get_tensor_rank(list_or_value)
        if rank == 0:
            return symbolic_helper._unsqueeze_helper(g, list_or_value, [0])
        if rank == 1:
            return list_or_value
        raise errors.SymbolicValueError(
            f"Rank must be 0 or 1, not {rank}", list_or_value
        )

    def get_const_value(list_or_value):
        if isinstance(list_or_value, (list, torch.Tensor)):
            if len(list_or_value) == 1:
                return list_or_value[0]
            return None
        return symbolic_helper._maybe_get_const(list_or_value, "i")

    # Check if slice is a no-op
    if (
        get_const_value(starts) == 0
        and get_const_value(ends) == _constants.INT64_MAX
        and (steps is None or get_const_value(steps) == 1)
    ):
        return input

    axes = to_slice_input(axes)
    starts = to_slice_input(starts, default_value=0)
    ends = to_slice_input(ends, default_value=_constants.INT64_MAX)
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    steps = to_slice_input(steps, default_value=1)
    return g.op("Slice", input, starts, ends, axes, steps)


@_onnx_symbolic("aten::slice")
@_beartype.beartype
def slice(g: jit_utils.GraphContext, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor
        dims, start, end, step = args
    elif len(args) == 3:
        # aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]
        start, end, step = args
        dims = [0]
    else:
        raise errors.SymbolicValueError("Unknown aten::slice signature", self)

    return symbolic_helper._slice_helper(
        g,
        self,
        axes=dims,
        starts=start,
        ends=end,
        steps=step,
    )


@_onnx_symbolic("aten::flip")
@symbolic_helper.parse_args("v", "is")
@_beartype.beartype
def flip(g: jit_utils.GraphContext, input, dims):
    return symbolic_helper._slice_helper(
        g,
        input,
        axes=dims,
        starts=[-1] * len(dims),
        ends=[-_constants.INT64_MAX] * len(dims),
        steps=[-1] * len(dims),
    )


@_onnx_symbolic("aten::fmod")
@_beartype.beartype
def fmod(g: jit_utils.GraphContext, input, other):
    return g.op("Mod", input, other, fmod_i=1)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
@_beartype.beartype
def embedding_bag(
    g: jit_utils.GraphContext,
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
    if scale_grad_by_freq and GLOBALS.export_training:
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


@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
@_beartype.beartype
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is a special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) == (0, 127):
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Quantize range (0, 127) not supported, requires opset 13 Clip",
            inputs,
        )
    if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
        raise errors.SymbolicValueError(
            f"For (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    scale = symbolic_helper._maybe_get_scalar(scale)
    if scale is None:
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Non-constant scale not supported",
            inputs,
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


@_onnx_symbolic("aten::isinf")
@_beartype.beartype
def isinf(g: jit_utils.GraphContext, input):
    return g.op("IsInf", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.DOUBLE))


@_onnx_symbolic("aten::isfinite")
@_beartype.beartype
def isfinite(g: jit_utils.GraphContext, input):
    inf_node = isinf(g, input)
    nan_node = opset9.isnan(g, input)
    return opset9.__not_(g, opset9.__or_(g, inf_node, nan_node))


@_onnx_symbolic("aten::quantize_per_tensor")
@_beartype.beartype
def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # TODO(justinchuby): Extract all the cast ops into a helper function.
    zero_point = g.op(
        "Cast", zero_point, to_i=_type_utils.JitScalarType(dtype).onnx_type()
    )
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return symbolic_helper.quantize_helper(g, input, scale, zero_point)


@_onnx_symbolic("aten::dequantize")
@_beartype.beartype
def dequantize(g: jit_utils.GraphContext, input):
    return symbolic_helper.dequantize_helper(g, input)[0]


@_onnx_symbolic("aten::nan_to_num")
@symbolic_helper.parse_args("v", "f", "f", "f")
@_beartype.beartype
def nan_to_num(g: jit_utils.GraphContext, input, nan, posinf, neginf):
    # Cannot create a int type tensor with inf/nan values, so we simply
    # return the original tensor
    if not symbolic_helper._is_fp(input):
        return input
    input_dtype = _type_utils.JitScalarType.from_value(input).dtype()
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
    # value representable by input's dtype.
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


# Quantized symbolics ---------------------------------------------------------
# https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
# Support starts from opset 10 because `DequantizeLinear` and `QuantizeLinear` were
# introduced in opset version 10.
@_onnx_symbolic("quantized::linear")
@_beartype.beartype
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::linear_relu")
@_beartype.beartype
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::add")
@_beartype.beartype
def quantized_add(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::add_relu")
@_beartype.beartype
def quantized_add_relu(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.add(g, x, y)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::mul")
@_beartype.beartype
def quantized_mul(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    output = opset9.mul(g, x, y)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::hardswish")
@_beartype.beartype
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.hardswish(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::sigmoid")
@_beartype.beartype
def quantized_sigmoid(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.sigmoid(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::leaky_relu")
@_beartype.beartype
def quantized_leaky_relu(
    g: jit_utils.GraphContext, x, negative_slope, inplace, op_scale, op_zero_point
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.leaky_relu(g, x, negative_slope, inplace)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::layer_norm")
@_beartype.beartype
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.layer_norm(g, x, normalized_shape, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::group_norm")
@_beartype.beartype
def quantized_group_norm(
    g: jit_utils.GraphContext,
    x,
    num_groups,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = opset9.group_norm(g, x, num_groups, weight, bias, eps, False)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v")
@_beartype.beartype
def quantized_instance_norm(
    g: jit_utils.GraphContext,
    q_input,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    input, _, _, _ = symbolic_helper.dequantize_helper(g, q_input)

    output = opset9.instance_norm(
        g, input, weight, bias, None, None, False, 0.0, eps, False
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv1d_relu")
@_beartype.beartype
def quantized_conv1d_relu(
    g: jit_utils.GraphContext,
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
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d_relu")
@_beartype.beartype
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,
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
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv3d_relu")
@_beartype.beartype
def quantized_conv3d_relu(
    g: jit_utils.GraphContext,
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
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv1d")
@_beartype.beartype
def quantized_conv1d(
    g: jit_utils.GraphContext,
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
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d")
@_beartype.beartype
def quantized_conv2d(
    g: jit_utils.GraphContext,
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
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv3d")
@_beartype.beartype
def quantized_conv3d(
    g: jit_utils.GraphContext,
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
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose1d")
@_beartype.beartype
def quantized_conv_transpose1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose2d")
@_beartype.beartype
def quantized_conv_transpose2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv_transpose3d")
@_beartype.beartype
def quantized_conv_transpose3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv_transpose3d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::cat")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def quantized_cat(
    g: jit_utils.GraphContext,
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
