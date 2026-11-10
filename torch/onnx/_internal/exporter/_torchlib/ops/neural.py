# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR neural translations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import normalization


if TYPE_CHECKING:
    from torch.onnx._internal.exporter._ir_builder import TranslationContext


aten = torch.ops.aten
_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808


def _adjust_attributes_of_avg_pool(
    op: TranslationContext,
    expand_size: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        pads = padding * expand_size
    else:
        pads = padding * 2

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads)


def _aten_avg_pool_onnx(
    op: TranslationContext,
    self: ir.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    ceil_mode: bool,
    count_include_pad: bool,
) -> ir.Value:
    self_rank_is_unbatched_rank = len(self.shape) == len(kernel_shape) + 1
    if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = op.Unsqueeze(self, [0])

    result = op.AveragePool(
        self,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    if self_rank_is_unbatched_rank:
        result = op.Squeeze(result, [0])

    return result


@onnx_impl(aten.avg_pool1d.default)
def aten_avg_pool1d(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0,),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> ir.Value:
    """avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor"""

    # Torch prefer to use single number x for kerne,stride,pad,dilation on both side implicitly
    # But ONNX needs pair number [x,y] to specify on each side explicitly
    # For pool3d, this number should be 3
    expand_size = 1

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        op, expand_size, kernel_size, stride, padding
    )

    return _aten_avg_pool_onnx(
        op, self, kernel_shape, strides, pads, ceil_mode, count_include_pad
    )


@onnx_impl(aten.avg_pool2d.default)
def aten_avg_pool2d(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> ir.Value:
    """avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor"""

    # Torch prefer to use single number x for kerne,stride,pad,dilation on both side implicitly
    # But ONNX needs pair number [x,y] to specify on each side explicitly
    # For pool3d, this number should be 3
    expand_size = 2

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        op, expand_size, kernel_size, stride, padding
    )

    # TODO: if want to support divisor_override argument, need to op.Mul(result, mask)
    # mask = [
    #    1, 2, 3, S,..3, 2, 1
    #    2, 4, 6, 2S, 6, 4, 2
    #    3, 6, 9, 3S, 9, 6, 3
    #    S, 2S,3S,SS,3S,2S, S
    #    3, 6, 9, 3S, 9, 6, 3
    #    2, 4, 6, 2S, 6, 4, 2
    #    1, 2, 3, S,..3, 2, 1
    # ]
    # S is stride size, in this case S=4,
    # S may dup lot of times according to the image size

    return _aten_avg_pool_onnx(
        op, self, kernel_shape, strides, pads, ceil_mode, count_include_pad
    )


@onnx_impl(aten.avg_pool3d.default)
def aten_avg_pool3d(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> ir.Value:
    """avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor"""

    # Torch prefer to use single number x for kerne,stride,pad,dilation on both side implicitly
    # But ONNX needs pair number [x,y] to specify on each side explicitly
    # For pool3d, this number should be 3
    expand_size = 3

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        op, expand_size, kernel_size, stride, padding
    )

    # TODO: if want to support divisor_override argument, need to op.Mul(result, mask)
    # mask = [
    #    1, 2, 3, S,..3, 2, 1
    #    2, 4, 6, 2S, 6, 4, 2
    #    3, 6, 9, 3S, 9, 6, 3
    #    S, 2S,3S,SS,3S,2S, S
    #    3, 6, 9, 3S, 9, 6, 3
    #    2, 4, 6, 2S, 6, 4, 2
    #    1, 2, 3, S,..3, 2, 1
    # ]
    # S is stride size, in this case S=4,
    # S may dup lot of times according to the image size

    return _aten_avg_pool_onnx(
        op, self, kernel_shape, strides, pads, ceil_mode, count_include_pad
    )


@onnx_impl(aten.celu.default)
def aten_celu(op: TranslationContext, self: ir.Value, alpha: float = 1.0) -> ir.Value:
    """celu(Tensor self, Scalar alpha=1.0) -> Tensor"""

    if self.dtype != ir.DataType.FLOAT:
        self_upcasted = op.Cast(self, to=ir.DataType.FLOAT)

        # op.Celu only support float32
        return op.Cast(op.Celu(self_upcasted, alpha=alpha), to=self.dtype)

    return op.Celu(self, alpha=alpha)


@onnx_impl(aten.col2im.default)
def aten_col2im(
    op: TranslationContext,
    self: ir.Value,
    output_size: ir.Value,
    kernel_size: ir.Value,
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
) -> ir.Value:
    """col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""

    # assert(len(output_size)==2) for ONNX
    # assert(len(kernel_size)==2) for ONNX
    # assert(len(dilation)==2) for ONNX
    # assert(len(stride)==2) for ONNX

    # The pads should be [w, x, y, z] for ONNX
    if len(padding) == 1:  # [w] -> [w, w, w, w]
        pads = padding * 4
    elif len(padding) == 2:  # [w, x] -> [w, x, w, x]
        pads = padding * 2
    else:  # assert len(padding) == 4, already [w, x, y, z]
        pads = padding

    return op.Col2Im(
        self,
        output_size,
        kernel_size,
        dilations=dilation,
        pads=pads,
        strides=stride,
    )


@onnx_impl(aten.cross_entropy_loss.default)
def aten_cross_entropy_loss(
    op: TranslationContext,
    self: ir.Value,
    target: ir.Value,
    weight: ir.Value | None = None,
    reduction: int = 1,  # default is 'mean'
    ignore_index: int = -100,
    label_smoothing: float = 0.0,  # this was ignored due to ONNX not support
) -> ir.Value:
    """cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor"""

    if reduction == 0:  # "none"
        result, _ = op.SoftmaxCrossEntropyLoss(
            self,
            target,
            weight,
            reduction="none",
            ignore_index=ignore_index,
            outputs=2,
        )
    elif reduction == 2:  # "sum"
        result, _ = op.SoftmaxCrossEntropyLoss(
            self,
            target,
            weight,
            reduction="sum",
            ignore_index=ignore_index,
            outputs=2,
        )
    else:  # "mean", default
        result, _ = op.SoftmaxCrossEntropyLoss(
            self,
            target,
            weight,
            reduction="mean",
            ignore_index=ignore_index,
            outputs=2,
        )

    return result


@onnx_impl(aten.elu.default)
def aten_elu(
    op: TranslationContext,
    self: ir.Value,
    alpha: float = 1.0,
    scale: float = 1.0,
    input_scale: float = 1.0,
) -> ir.Value:
    """elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"""

    input_scale = op.CastLike(input_scale, self)
    scale = op.CastLike(scale, self)
    self = op.Mul(self, input_scale)
    return op.Mul(op.Elu(self, alpha=alpha), scale)


@onnx_impl(aten.gelu.default, opset_introduced=18)
def aten_gelu(
    op: TranslationContext, self: ir.Value, approximate: str = "none"
) -> ir.Value:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    if approximate == "tanh":
        result = _aten_gelu_approximate_tanh(op, self)
    else:
        result = _aten_gelu_approximate_none(op, self)
    return result


def _aten_gelu_approximate_none(op: TranslationContext, self: ir.Value) -> ir.Value:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""
    builder = op.builder

    # GELU(x) = 0.5 * x * [1 + ERF(x/sqrt(2)]
    inner = op.Div(self, builder.constant(1.4142135623730951, dtype=self.dtype))
    erf = op.Erf(inner)
    inner = op.Add(erf, builder.constant(1, dtype=self.dtype))
    inner = op.Mul(builder.constant(0.5, dtype=self.dtype), inner)
    result = op.Mul(self, inner)
    return result


def _aten_gelu_approximate_tanh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""
    builder = op.builder

    # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
    cubed = op.Pow(self, builder.constant(3, dtype=self.dtype))
    inner = op.Mul(builder.constant(0.044715, dtype=self.dtype), cubed)
    inner = op.Add(self, inner)
    # math.sqrt(2.0/math.pi) = 0.7978845608028654
    sqrt_two_over_pi = builder.constant(0.7978845608028654, dtype=self.dtype)
    inner = op.Mul(sqrt_two_over_pi, inner)
    inner = op.Tanh(inner)
    inner = op.Add(inner, builder.constant(1, dtype=self.dtype))
    inner = op.Mul(builder.constant(0.5, dtype=self.dtype), inner)
    result = op.Mul(self, inner)
    return result


@onnx_impl(aten.glu.default)
def aten_glu(op: TranslationContext, self: ir.Value, dim: int = -1) -> ir.Value:
    """glu(Tensor self, int dim=-1) -> Tensor"""

    first, second = op.Split(self, axis=dim, num_outputs=2, outputs=2)
    result = op.Mul(first, op.Sigmoid(second))
    return result


@onnx_impl(aten.group_norm.default, opset_introduced=18)
def aten_group_norm(
    op: TranslationContext,
    input: ir.Value,
    num_groups: int,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> ir.Value:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    return normalization.aten_native_group_norm(
        op, input, weight, bias, group=num_groups, eps=eps
    )[0]


@onnx_impl(aten.hardsigmoid.default)
def aten_hardsigmoid(op: TranslationContext, self: ir.Value) -> ir.Value:
    """hardsigmoid(Tensor self) -> Tensor"""

    return op.HardSigmoid(self, alpha=1 / 6, beta=1 / 2)


@onnx_impl(aten.hardswish.default)
def aten_hardswish(op: TranslationContext, self: ir.Value) -> ir.Value:
    """hardswish(Tensor self) -> Tensor"""

    return op.HardSwish(self)


@onnx_impl(aten.hardtanh.default)
def aten_hardtanh(
    op: TranslationContext, self: ir.Value, min_val: float = -1.0, max_val: float = 1.0
) -> ir.Value:
    """hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"""

    return op.Clip(self, min_val, max_val)


@onnx_impl(aten.hardtanh_backward.default)
def aten_hardtanh_backward(
    op: TranslationContext,
    grad_output: ir.Value,
    self: ir.Value,
    min_val: float,
    max_val: float,
) -> ir.Value:
    """hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor"""

    max_mask = op.Where(op.Greater(self, max_val), 0.0, 1.0)
    min_mask = op.Where(op.Less(self, min_val), 0.0, 1.0)
    return op.Mul(op.Mul(grad_output, max_mask), min_mask)


def _get_im2col_indices_along_dim(
    op: TranslationContext,
    input_d: ir.Value,
    kernel_size_d: int,
    dilation_d: int,
    padding_d: int,
    stride_d: int,
):
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    blocks_d = op.Add(input_d, (padding_d * 2) - (dilation_d * (kernel_size_d - 1)))

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = op.Range(0, blocks_d, stride_d)
    blocks_d_indices = op.Unsqueeze(blocks_d_indices, [0])

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = op.Range(0, kernel_size_d * dilation_d, dilation_d)
    kernel_mask = op.Unsqueeze(kernel_grid, [1])

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    block_mask = op.Add(blocks_d_indices, kernel_mask)

    return block_mask


def _get_im2col_padded_input(op: TranslationContext, input, padding_h, padding_w):
    builder = op.builder
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format: (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = op.Concat(
        builder.constant([0, 0], dtype=ir.DataType.INT64),
        op.Unsqueeze(padding_h, [0]),
        op.Unsqueeze(padding_w, [0]),
        builder.constant([0, 0], dtype=ir.DataType.INT64),
        op.Unsqueeze(padding_h, [0]),
        op.Unsqueeze(padding_w, [0]),
        axis=0,
    )
    return op.Pad(input, pad)


def _get_im2col_output_shape(op: TranslationContext, input, kernel_h, kernel_w):
    builder = op.builder
    input_shape = op.Shape(input)
    batch_dim = op.Gather(input_shape, 0, axis=0)
    channel_dim = op.Gather(input_shape, 1, axis=0)
    channel_unfolded = op.Mul(channel_dim, kernel_h * kernel_w)

    return op.Concat(
        op.Unsqueeze(batch_dim, [0]),
        op.Unsqueeze(channel_unfolded, [0]),
        builder.constant([-1], dtype=ir.DataType.INT64),
        axis=0,
    )


@onnx_impl(aten.im2col.default)
def aten_im2col(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
) -> ir.Value:
    """im2col(Tensor self, int[2] kernel_size, int[2] dilation=1, int[2] padding=0, int[2] stride=1) -> Tensor"""

    input_shape = op.Shape(self)
    input_h = op.Gather(input_shape, 2, axis=0)
    input_w = op.Gather(input_shape, 3, axis=0)

    if not isinstance(kernel_size, Sequence):
        kernel_size = (kernel_size, kernel_size)
    kernel_sizes = list(kernel_size)

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation)
    dilations = list(dilation)

    if not isinstance(padding, Sequence):
        padding = (padding, padding)
    pads = list(padding)

    if isinstance(stride, int):
        stride = (stride, stride)
    strides = list(stride)

    stride_h, stride_w = strides[0], strides[1]
    padding_h, padding_w = pads[0], pads[1]
    dilation_h, dilation_w = dilations[0], dilations[1]
    kernel_h, kernel_w = kernel_sizes[0], kernel_sizes[1]

    blocks_row_indices = _get_im2col_indices_along_dim(
        op, input_h, kernel_h, dilation_h, padding_h, stride_h
    )
    blocks_col_indices = _get_im2col_indices_along_dim(
        op, input_w, kernel_w, dilation_w, padding_w, stride_w
    )

    output_shape = _get_im2col_output_shape(op, self, kernel_h, kernel_w)
    padded_input = _get_im2col_padded_input(op, self, padding_h, padding_w)

    # For a 4D matrix of size (1, 1, 3, 3) as below with kernel_size=2, stride=1, and dilation=1
    # [[[[1., 2., 3.,],
    #    [4., 5., 6.,],
    #    [7., 8., 9.,]]]]
    # First gather indices along rows (dim=2) with blocks_row_indices = [[0,1], [1,2]] to get:
    # [[[[[1., 2., 3.],
    #     [4., 5., 6.]],
    #    [[4., 5., 6.],
    #     [7., 8., 9.]]]]]
    # And then gather along cols (dim=4) with blocks_row_indices = [[0,1], [1,2]] to get:
    # [[[[[[1., 2.],
    #      [4., 5.]],
    #     [[2., 3.],
    #      [5., 6]]],
    #    [[[4., 5.],
    #      [7., 8.]],
    #     [[5., 6.],
    #      [8., 9.]]]]]]
    # Transpose dims 3 (depth) and 4 (rows), and then reshape to output shape (1, 1, 4, 4) to get:
    #  [[[1., 2., 4., 5.],
    #    [2., 3., 5., 6.],
    #    [4., 5., 7., 8.],
    #    [5., 6., 8., 9.]]]
    output = op.Gather(padded_input, blocks_row_indices, axis=2)
    output = op.Gather(output, blocks_col_indices, axis=4)
    output = op.Transpose(output, perm=[0, 1, 2, 4, 3, 5])
    return op.Reshape(output, output_shape)


@onnx_impl(aten.leaky_relu.default)
def aten_leaky_relu(
    op: TranslationContext, self: ir.Value, negative_slope: float = 0.01
) -> ir.Value:
    """leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"""

    return op.LeakyRelu(self, alpha=negative_slope)


@onnx_impl(aten.linear.default)
def aten_linear(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
) -> ir.Value:
    """linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"""

    if len(input.shape) == 2 and len(weight.shape) == 2:
        # Use Gemm for the rank 2 input
        return op.Gemm(input, weight, bias, transB=True)
    if len(weight.shape) == 1:
        # In rare cases the weight can be 1d. aten::linear does not support a
        # bias with a 1d weight (mat2 must be a matrix). eager linear contracts
        # the last dim away in this case, so squeeze the dim added for MatMul
        # back off.
        if bias is not None:
            raise NotImplementedError(
                "aten::linear with 1D weight and bias is not supported. "
                "Omit bias or supply a 2D weight."
            )

        weight_transposed = op.Unsqueeze(weight, [1])
        return op.Squeeze(op.MatMul(input, weight_transposed), [-1])
    if len(weight.shape) != 2:
        raise ValueError(
            f"aten::linear expects a 1D or 2D weight, got shape {weight.shape}. "
            "Supply a weight with at most two dimensions."
        )
    weight_transposed = op.Transpose(weight, perm=[1, 0])
    mul = op.MatMul(input, weight_transposed)
    if bias is None:
        return mul
    return op.Add(mul, bias)


@onnx_impl(aten.log_sigmoid.default)
def aten_log_sigmoid(op: TranslationContext, self: ir.Value) -> ir.Value:
    """log_sigmoid(Tensor self) -> Tensor"""

    return op.Log(op.Sigmoid(self))


@onnx_impl(aten.max_pool1d.default)
def aten_max_pool1d(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> ir.Value:
    """max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 1

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        op, expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        op, self, kernel_shape, strides, pads, dilations, ceil_mode, 2
    )


@onnx_impl(("aten::max_pool1d_with_indices",))
def aten_max_pool1d_with_indices(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 1

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        op, expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        op,
        self,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        2,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
    )


def _adjust_attributes_of_max_pool(
    op: TranslationContext,
    expand_size: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    if isinstance(dilation, int):
        dilations = [dilation] * expand_size
    else:
        dilations = dilation

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    # NOTE: expand_size is the dimension of pooling kernel,
    # ONNX needs begin and end padding so we need to double the padding

    # NOTE: expand size prevents padding from being a single int in
    # multiple dimension cases
    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads, dilations)


@onnx_impl(aten.max_pool2d.default)
def aten_max_pool2d(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> ir.Value:
    """max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a pair of number [x,y] on each side explicitly.
    expand_size = 2

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        op, expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        op, self, kernel_shape, strides, pads, dilations, ceil_mode, 3
    )


def _aten_max_pool_onnx(
    op: TranslationContext,
    self: ir.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> ir.Value:
    self_rank_is_unbatched_rank = len(self.shape) == unbatched_rank
    if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = op.Unsqueeze(self, [0])

    pool_result, _ = op.MaxPool(
        self,
        ceil_mode=ceil_mode,
        dilations=dilations,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        outputs=2,
    )

    if self_rank_is_unbatched_rank:
        pool_result = op.Squeeze(pool_result, [0])

    return pool_result


@onnx_impl(aten.max_pool3d.default)
def aten_max_pool3d(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> ir.Value:
    """max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 3

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        op, expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        op, self, kernel_shape, strides, pads, dilations, ceil_mode, 4
    )


@onnx_impl(aten.max_pool2d_with_indices.default)
def aten_max_pool2d_with_indices(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a pair of number [x,y] on each side explicitly.
    expand_size = 2

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        op, expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        op,
        self,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        3,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
    )


@onnx_impl(aten.max_pool3d_with_indices.default)
def aten_max_pool3d_with_indices(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 3

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        op, expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        op,
        self,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        4,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
    )


def _aten_max_pool_with_indices_onnx(
    op: TranslationContext,
    self: ir.Value,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
) -> tuple[ir.Value, ir.Value]:
    builder = op.builder
    self_rank_is_unbatched_rank = len(self.shape) == unbatched_rank
    if self_rank_is_unbatched_rank:
        self = op.Unsqueeze(self, [0])

    pool_result, indices = op.MaxPool(
        self,
        ceil_mode=ceil_mode,
        dilations=dilation,
        kernel_shape=kernel_size,
        pads=padding,
        strides=stride,
        outputs=2,
    )

    # Simple but hacky way to get flattened indices values
    # to be used to convert the indices values to non-flattened.
    # In ONNX the indices are computed as a flatten 1-D tensor,
    # so the values in indices are in [0, N x C x D1 x ... x Dn).
    # To convert the indices to the same format used by PyTorch,
    # we first execute a maxpool with a kernel and stride of 1 on the same input.
    # This will result in a tensor of indices in which each index will have it's own value.
    # Using this tensor as a reference, we extract the first index of each axis and subtract
    # it from each index of this axis in the indices to convert.
    # This step will result in a tensor where each dimension has values of indices within
    # the dimension it is in.
    # For Maxpool1d(kernel=1,stride=1,return_indices=True), with the input torch.ones(1,2,2).
    # The computed indices are the following:
    # output indices pytorch :
    #     [[0,1],
    #      [0,1]]
    # output indices onnx:
    #     [[0,1],
    #      [2,3]]
    # The purpose was to convert the indices from one format to the other to be able to match the results.
    # So flattened_indices will have the value of each index and will be equal to :
    #     [[0,1],
    #     [2,3]]
    # Then call Slice to get the first value of each line (so 0 and 2).
    # And the subtraction executes :
    #     [[0-0,1-0],
    #     [2-2,3-2]]
    # So indices results to the expected output which is :
    #     [[0,1],
    #     [0,1]]
    # For more information :
    # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
    _, flatten_indices = op.MaxPool(
        self,
        dilations=dilation,
        kernel_shape=n_dims_one,
        strides=n_dims_one,
        outputs=2,
    )

    ends = builder.constant(n_dims_one, dtype=ir.DataType.INT64)
    starts = builder.constant(n_dims_zero, dtype=ir.DataType.INT64)
    axes = builder.constant(n_dims_axes, dtype=ir.DataType.INT64)

    delta = op.Slice(flatten_indices, starts, ends, axes)
    indices = op.Sub(indices, delta)

    if self_rank_is_unbatched_rank:
        pool_result = op.Squeeze(pool_result, [0])
        indices = op.Squeeze(indices, [0])

    return (pool_result, indices)


@onnx_impl(aten.mish.default)
def aten_mish(op: TranslationContext, self: ir.Value) -> ir.Value:
    """mish(Tensor self) -> Tensor"""

    return op.Mish(self)


@onnx_impl(aten.mse_loss.default)
def aten_mse_loss(
    op: TranslationContext, self: ir.Value, target: ir.Value, reduction: int = 1
) -> ir.Value:
    """mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"""
    # FIXME: When reduction=0, the shape(result) will be different than other case
    result = op.Mul(
        op.Sub(self, target),
        op.Sub(self, target),
    )
    if reduction == 1:  # mean
        result = op.ReduceMean(result, keepdims=False)
    if reduction == 2:  # sum
        result = op.ReduceSum(result, keepdims=False)

    return result


@onnx_impl(aten.nll_loss.default)
def aten_nll_loss(
    op: TranslationContext,
    self: ir.Value,
    target: ir.Value,
    weight: ir.Value | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
) -> ir.Value:
    """nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor"""

    self_rank_is_1 = len(self.shape) == 1
    if self_rank_is_1:  # self rank should be at least 2
        self = op.Unsqueeze(self, [0])

    rank_target = len(target.shape)
    if rank_target == 0:  # target rank should be at least 1
        target = op.Unsqueeze(target, [0])

    if reduction == 0:
        reduction_str = "none"
    elif reduction == 1:
        reduction_str = "mean"
    else:  # assert reduction == 2
        reduction_str = "sum"

    result = op.NegativeLogLikelihoodLoss(
        self,
        target,
        weight,
        ignore_index=ignore_index,
        reduction=reduction_str,
    )

    if self_rank_is_1:
        result = op.Squeeze(result)

    return result


@onnx_impl(aten.nll_loss_forward.default)
def aten_nll_loss_forward(
    op: TranslationContext,
    self: ir.Value,
    target: ir.Value,
    weight: ir.Value | None,
    reduction: int,
    ignore_index: int,
) -> tuple[ir.Value, ir.Value]:
    """nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""

    output = aten_nll_loss(op, self, target, weight, reduction, ignore_index)
    # FIXME: Fake a total_weight tensor for now. It should be different based on weight, reduction and ignore_index
    if weight is None:
        total_weight = op.CastLike(op.Size(output), self)
    else:
        total_weight = op.CastLike(op.Size(output), weight)
    return output, total_weight


def _process_padding(
    op: TranslationContext, padding: Sequence[ir.Value | int], rank: int
) -> ir.Value:
    """Convert PyTorch padding for ONNX Pad."""
    builder = op.builder
    if not isinstance(padding, (list, tuple)):
        raise ValueError(
            f"ONNX padding expects a list or tuple, got {type(padding).__name__}. "
            "Supply the before/after padding for each padded dimension."
        )
    if all(isinstance(pad, int) for pad in padding):
        paddings = padding
        zeros = [0] * (rank * 2 - len(paddings))
        paddings = [*paddings, *zeros]
        paddings = paddings[-2::-2] + paddings[-1::-2]
        return op.Constant(value=ir.tensor(paddings, dtype=ir.DataType.INT64))
    else:
        paddings = []
        for pad in padding:
            if isinstance(pad, int):
                paddings.append(builder.constant([pad], dtype=ir.DataType.INT64))
            else:
                # Dynamic value
                paddings.append(op.Reshape(pad, [-1]))
        # Create a series of 1d zero tensors
        zero = builder.constant([0], dtype=ir.DataType.INT64)
        zeros = [zero] * (rank * 2 - len(paddings))
        paddings = [*paddings, *zeros]
        # Interleave the padding values
        paddings = paddings[-2::-2] + paddings[-1::-2]
        return op.Concat(*paddings, axis=0)
