# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR neural translations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import (
    common as common_ops,
    normalization,
)
from torch.onnx._internal.exporter._torchlib.ops.nn import (
    _attention_repeat_kv_for_group_query,
    _attention_scale,
)


aten = torch.ops.aten
_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808


def _adjust_attributes_of_avg_pool(
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
    self: ir.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    ceil_mode: bool,
    count_include_pad: bool,
) -> ir.Value:
    builder = current_builder()
    self_rank_is_unbatched_rank = len(self.shape) == len(kernel_shape) + 1
    if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = builder.call("Unsqueeze", self, [0])[0]

    result = builder.call(
        "AveragePool",
        self,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )[0]

    if self_rank_is_unbatched_rank:
        result = builder.call("Squeeze", result, [0])[0]

    return result


@onnx_impl(aten.avg_pool1d.default)
def aten_avg_pool1d(
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
        expand_size, kernel_size, stride, padding
    )

    return _aten_avg_pool_onnx(
        self, kernel_shape, strides, pads, ceil_mode, count_include_pad
    )


@onnx_impl(aten.avg_pool2d.default)
def aten_avg_pool2d(
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
        expand_size, kernel_size, stride, padding
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
        self, kernel_shape, strides, pads, ceil_mode, count_include_pad
    )


@onnx_impl(aten.avg_pool3d.default)
def aten_avg_pool3d(
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
        expand_size, kernel_size, stride, padding
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
        self, kernel_shape, strides, pads, ceil_mode, count_include_pad
    )


@onnx_impl(aten.celu.default)
def aten_celu(self: ir.Value, alpha: float = 1.0) -> ir.Value:
    """celu(Tensor self, Scalar alpha=1.0) -> Tensor"""
    builder = current_builder()

    if self.dtype != ir.DataType.FLOAT:
        self_upcasted = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]

        # op.Celu only support float32
        return builder.call(
            "Cast", builder.call("Celu", self_upcasted, alpha=alpha)[0], to=self.dtype
        )[0]

    return builder.call("Celu", self, alpha=alpha)[0]


@onnx_impl(aten.col2im.default)
def aten_col2im(
    self: ir.Value,
    output_size: ir.Value,
    kernel_size: ir.Value,
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
) -> ir.Value:
    """col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""
    builder = current_builder()

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

    return builder.call(
        "Col2Im",
        self,
        output_size,
        kernel_size,
        dilations=dilation,
        pads=pads,
        strides=stride,
    )[0]


@onnx_impl(aten.cross_entropy_loss.default)
def aten_cross_entropy_loss(
    self: ir.Value,
    target: ir.Value,
    weight: ir.Value | None = None,
    reduction: int = 1,  # default is 'mean'
    ignore_index: int = -100,
    label_smoothing: float = 0.0,  # this was ignored due to ONNX not support
) -> ir.Value:
    """cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor"""
    builder = current_builder()

    if reduction == 0:  # "none"
        result, _ = builder.call(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction="none",
            ignore_index=ignore_index,
            _num_outputs=2,
        )
    elif reduction == 2:  # "sum"
        result, _ = builder.call(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction="sum",
            ignore_index=ignore_index,
            _num_outputs=2,
        )
    else:  # "mean", default
        result, _ = builder.call(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction="mean",
            ignore_index=ignore_index,
            _num_outputs=2,
        )

    return result


@onnx_impl(aten.elu.default)
def aten_elu(
    self: ir.Value,
    alpha: float = 1.0,
    scale: float = 1.0,
    input_scale: float = 1.0,
) -> ir.Value:
    """elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"""
    builder = current_builder()

    input_scale = builder.call("CastLike", input_scale, self)[0]
    scale = builder.call("CastLike", scale, self)[0]
    self = builder.call("Mul", self, input_scale)[0]
    return builder.call("Mul", builder.call("Elu", self, alpha=alpha)[0], scale)[0]


@onnx_impl(aten.gelu.default, opset_introduced=18)
def aten_gelu(self: ir.Value, approximate: str = "none") -> ir.Value:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    if approximate == "tanh":
        result = _aten_gelu_approximate_tanh(self)
    else:
        result = _aten_gelu_approximate_none(self)
    return result


def _aten_gelu_approximate_none(self: ir.Value) -> ir.Value:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""
    builder = current_builder()

    # GELU(x) = 0.5 * x * [1 + ERF(x/sqrt(2)]
    inner = builder.call(
        "Div", self, builder.constant(1.4142135623730951, dtype=self.dtype)
    )[0]
    erf = builder.call("Erf", inner)[0]
    inner = builder.call("Add", erf, builder.constant(1, dtype=self.dtype))[0]
    inner = builder.call("Mul", builder.constant(0.5, dtype=self.dtype), inner)[0]
    result = builder.call("Mul", self, inner)[0]
    return result


def _aten_gelu_approximate_tanh(self: ir.Value) -> ir.Value:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""
    builder = current_builder()

    # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
    cubed = builder.call("Pow", self, builder.constant(3, dtype=self.dtype))[0]
    inner = builder.call("Mul", builder.constant(0.044715, dtype=self.dtype), cubed)[0]
    inner = builder.call("Add", self, inner)[0]
    # math.sqrt(2.0/math.pi) = 0.7978845608028654
    sqrt_two_over_pi = builder.constant(0.7978845608028654, dtype=self.dtype)
    inner = builder.call("Mul", sqrt_two_over_pi, inner)[0]
    inner = builder.call("Tanh", inner)[0]
    inner = builder.call("Add", inner, builder.constant(1, dtype=self.dtype))[0]
    inner = builder.call("Mul", builder.constant(0.5, dtype=self.dtype), inner)[0]
    result = builder.call("Mul", self, inner)[0]
    return result


@onnx_impl(aten.glu.default)
def aten_glu(self: ir.Value, dim: int = -1) -> ir.Value:
    """glu(Tensor self, int dim=-1) -> Tensor"""
    builder = current_builder()

    first, second = builder.call("Split", self, axis=dim, num_outputs=2, _num_outputs=2)
    result = builder.call("Mul", first, builder.call("Sigmoid", second)[0])[0]
    return result


@onnx_impl(aten.group_norm.default, opset_introduced=18)
def aten_group_norm(
    input: ir.Value,
    num_groups: int,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> ir.Value:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    return normalization.aten_native_group_norm(
        input, weight, bias, group=num_groups, eps=eps
    )[0]


@onnx_impl(aten.hardsigmoid.default)
def aten_hardsigmoid(self: ir.Value) -> ir.Value:
    """hardsigmoid(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("HardSigmoid", self, alpha=1 / 6, beta=1 / 2)[0]


@onnx_impl(aten.hardswish.default)
def aten_hardswish(self: ir.Value) -> ir.Value:
    """hardswish(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("HardSwish", self)[0]


@onnx_impl(aten.hardtanh.default)
def aten_hardtanh(
    self: ir.Value, min_val: float = -1.0, max_val: float = 1.0
) -> ir.Value:
    """hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"""
    builder = current_builder()

    return builder.call("Clip", self, min_val, max_val)[0]


@onnx_impl(aten.hardtanh_backward.default)
def aten_hardtanh_backward(
    grad_output: ir.Value, self: ir.Value, min_val: float, max_val: float
) -> ir.Value:
    """hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor"""
    builder = current_builder()

    max_mask = builder.call(
        "Where", builder.call("Greater", self, max_val)[0], 0.0, 1.0
    )[0]
    min_mask = builder.call("Where", builder.call("Less", self, min_val)[0], 0.0, 1.0)[
        0
    ]
    return builder.call("Mul", builder.call("Mul", grad_output, max_mask)[0], min_mask)[
        0
    ]


def _get_im2col_indices_along_dim(
    input_d: ir.Value,
    kernel_size_d: int,
    dilation_d: int,
    padding_d: int,
    stride_d: int,
):
    builder = current_builder()
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    blocks_d = builder.call(
        "Add", input_d, (padding_d * 2) - (dilation_d * (kernel_size_d - 1))
    )[0]

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = builder.call("Range", 0, blocks_d, stride_d)[0]
    blocks_d_indices = builder.call("Unsqueeze", blocks_d_indices, [0])[0]

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = builder.call("Range", 0, kernel_size_d * dilation_d, dilation_d)[0]
    kernel_mask = builder.call("Unsqueeze", kernel_grid, [1])[0]

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    block_mask = builder.call("Add", blocks_d_indices, kernel_mask)[0]

    return block_mask


def _get_im2col_padded_input(input, padding_h, padding_w):
    builder = current_builder()
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format: (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = builder.call(
        "Concat",
        builder.constant([0, 0], dtype=ir.DataType.INT64),
        builder.call("Unsqueeze", padding_h, [0])[0],
        builder.call("Unsqueeze", padding_w, [0])[0],
        builder.constant([0, 0], dtype=ir.DataType.INT64),
        builder.call("Unsqueeze", padding_h, [0])[0],
        builder.call("Unsqueeze", padding_w, [0])[0],
        axis=0,
    )[0]
    return builder.call("Pad", input, pad)[0]


def _get_im2col_output_shape(input, kernel_h, kernel_w):
    builder = current_builder()
    input_shape = builder.call("Shape", input)[0]
    batch_dim = builder.call("Gather", input_shape, 0, axis=0)[0]
    channel_dim = builder.call("Gather", input_shape, 1, axis=0)[0]
    channel_unfolded = builder.call("Mul", channel_dim, kernel_h * kernel_w)[0]

    return builder.call(
        "Concat",
        builder.call("Unsqueeze", batch_dim, [0])[0],
        builder.call("Unsqueeze", channel_unfolded, [0])[0],
        builder.constant([-1], dtype=ir.DataType.INT64),
        axis=0,
    )[0]


@onnx_impl(aten.im2col.default)
def aten_im2col(
    self: ir.Value,
    kernel_size: Sequence[int],
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
) -> ir.Value:
    """im2col(Tensor self, int[2] kernel_size, int[2] dilation=1, int[2] padding=0, int[2] stride=1) -> Tensor"""
    builder = current_builder()

    input_shape = builder.call("Shape", self)[0]
    input_h = builder.call("Gather", input_shape, 2, axis=0)[0]
    input_w = builder.call("Gather", input_shape, 3, axis=0)[0]

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
        input_h, kernel_h, dilation_h, padding_h, stride_h
    )
    blocks_col_indices = _get_im2col_indices_along_dim(
        input_w, kernel_w, dilation_w, padding_w, stride_w
    )

    output_shape = _get_im2col_output_shape(self, kernel_h, kernel_w)
    padded_input = _get_im2col_padded_input(self, padding_h, padding_w)

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
    output = builder.call("Gather", padded_input, blocks_row_indices, axis=2)[0]
    output = builder.call("Gather", output, blocks_col_indices, axis=4)[0]
    output = builder.call("Transpose", output, perm=[0, 1, 2, 4, 3, 5])[0]
    return builder.call("Reshape", output, output_shape)[0]


@onnx_impl(aten.leaky_relu.default)
def aten_leaky_relu(self: ir.Value, negative_slope: float = 0.01) -> ir.Value:
    """leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"""
    builder = current_builder()

    return builder.call("LeakyRelu", self, alpha=negative_slope)[0]


@onnx_impl(aten.linear.default)
def aten_linear(
    input: ir.Value, weight: ir.Value, bias: ir.Value | None = None
) -> ir.Value:
    """linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"""
    builder = current_builder()

    if len(input.shape) == 2 and len(weight.shape) == 2:
        # Use Gemm for the rank 2 input
        return builder.call("Gemm", input, weight, bias, transB=True)[0]
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

        weight_transposed = builder.call("Unsqueeze", weight, [1])[0]
        return builder.call(
            "Squeeze", builder.call("MatMul", input, weight_transposed)[0], [-1]
        )[0]
    if len(weight.shape) != 2:
        raise ValueError(
            f"aten::linear expects a 1D or 2D weight, got shape {weight.shape}. "
            "Supply a weight with at most two dimensions."
        )
    weight_transposed = builder.call("Transpose", weight, perm=[1, 0])[0]
    mul = builder.call("MatMul", input, weight_transposed)[0]
    if bias is None:
        return mul
    return builder.call("Add", mul, bias)[0]


@onnx_impl(aten.log_sigmoid.default)
def aten_log_sigmoid(self: ir.Value) -> ir.Value:
    """log_sigmoid(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Log", builder.call("Sigmoid", self)[0])[0]


@onnx_impl(aten.max_pool1d.default)
def aten_max_pool1d(
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
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        self, kernel_shape, strides, pads, dilations, ceil_mode, 2
    )


@onnx_impl(("aten::max_pool1d_with_indices",))
def aten_max_pool1d_with_indices(
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
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
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
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        self, kernel_shape, strides, pads, dilations, ceil_mode, 3
    )


def _aten_max_pool_onnx(
    self: ir.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> ir.Value:
    builder = current_builder()
    self_rank_is_unbatched_rank = len(self.shape) == unbatched_rank
    if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = builder.call("Unsqueeze", self, [0])[0]

    pool_result, _ = builder.call(
        "MaxPool",
        self,
        ceil_mode=ceil_mode,
        dilations=dilations,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        _num_outputs=2,
    )

    if self_rank_is_unbatched_rank:
        pool_result = builder.call("Squeeze", pool_result, [0])[0]

    return pool_result


@onnx_impl(aten.max_pool3d.default)
def aten_max_pool3d(
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
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        self, kernel_shape, strides, pads, dilations, ceil_mode, 4
    )


@onnx_impl(aten.max_pool2d_with_indices.default)
def aten_max_pool2d_with_indices(
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
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
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
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
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
    builder = current_builder()
    self_rank_is_unbatched_rank = len(self.shape) == unbatched_rank
    if self_rank_is_unbatched_rank:
        self = builder.call("Unsqueeze", self, [0])[0]

    pool_result, indices = builder.call(
        "MaxPool",
        self,
        ceil_mode=ceil_mode,
        dilations=dilation,
        kernel_shape=kernel_size,
        pads=padding,
        strides=stride,
        _num_outputs=2,
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
    _, flatten_indices = builder.call(
        "MaxPool",
        self,
        dilations=dilation,
        kernel_shape=n_dims_one,
        strides=n_dims_one,
        _num_outputs=2,
    )

    ends = builder.constant(n_dims_one, dtype=ir.DataType.INT64)
    starts = builder.constant(n_dims_zero, dtype=ir.DataType.INT64)
    axes = builder.constant(n_dims_axes, dtype=ir.DataType.INT64)

    delta = builder.call("Slice", flatten_indices, starts, ends, axes)[0]
    indices = builder.call("Sub", indices, delta)[0]

    if self_rank_is_unbatched_rank:
        pool_result = builder.call("Squeeze", pool_result, [0])[0]
        indices = builder.call("Squeeze", indices, [0])[0]

    return (pool_result, indices)


@onnx_impl(aten.mish.default)
def aten_mish(self: ir.Value) -> ir.Value:
    """mish(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Mish", self)[0]


@onnx_impl(aten.mse_loss.default)
def aten_mse_loss(self: ir.Value, target: ir.Value, reduction: int = 1) -> ir.Value:
    """mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"""
    builder = current_builder()
    # FIXME: When reduction=0, the shape(result) will be different than other case
    result = builder.call(
        "Mul",
        builder.call("Sub", self, target)[0],
        builder.call("Sub", self, target)[0],
    )[0]
    if reduction == 1:  # mean
        result = builder.call("ReduceMean", result, keepdims=False)[0]
    if reduction == 2:  # sum
        result = builder.call("ReduceSum", result, keepdims=False)[0]

    return result


@onnx_impl(aten.nll_loss.default)
def aten_nll_loss(
    self: ir.Value,
    target: ir.Value,
    weight: ir.Value | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
) -> ir.Value:
    """nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor"""
    builder = current_builder()

    self_rank_is_1 = len(self.shape) == 1
    if self_rank_is_1:  # self rank should be at least 2
        self = builder.call("Unsqueeze", self, [0])[0]

    rank_target = len(target.shape)
    if rank_target == 0:  # target rank should be at least 1
        target = builder.call("Unsqueeze", target, [0])[0]

    if reduction == 0:
        reduction_str = "none"
    elif reduction == 1:
        reduction_str = "mean"
    else:  # assert reduction == 2
        reduction_str = "sum"

    result = builder.call(
        "NegativeLogLikelihoodLoss",
        self,
        target,
        weight,
        ignore_index=ignore_index,
        reduction=reduction_str,
    )[0]

    if self_rank_is_1:
        result = builder.call("Squeeze", result)[0]

    return result


@onnx_impl(aten.nll_loss_forward.default)
def aten_nll_loss_forward(
    self: ir.Value,
    target: ir.Value,
    weight: ir.Value | None,
    reduction: int,
    ignore_index: int,
) -> tuple[ir.Value, ir.Value]:
    """nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""
    builder = current_builder()

    output = aten_nll_loss(self, target, weight, reduction, ignore_index)
    # FIXME: Fake a total_weight tensor for now. It should be different based on weight, reduction and ignore_index
    if weight is None:
        total_weight = builder.call("CastLike", builder.call("Size", output)[0], self)[
            0
        ]
    else:
        total_weight = builder.call(
            "CastLike", builder.call("Size", output)[0], weight
        )[0]
    return output, total_weight


def _process_padding(padding: Sequence[ir.Value | int], rank: int) -> ir.Value:
    """Convert PyTorch padding for ONNX Pad."""
    builder = current_builder()
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
        return builder.call(
            "Constant", value=ir.tensor(paddings, dtype=ir.DataType.INT64)
        )[0]
    else:
        paddings = []
        for pad in padding:
            if isinstance(pad, int):
                paddings.append(builder.constant([pad], dtype=ir.DataType.INT64))
            else:
                # Dynamic value
                paddings.append(builder.call("Reshape", pad, [-1])[0])
        # Create a series of 1d zero tensors
        zero = builder.constant([0], dtype=ir.DataType.INT64)
        zeros = [zero] * (rank * 2 - len(paddings))
        paddings = [*paddings, *zeros]
        # Interleave the padding values
        paddings = paddings[-2::-2] + paddings[-1::-2]
        return builder.call("Concat", *paddings, axis=0)[0]


@onnx_impl(aten.pad.default)
def aten_pad(
    self: ir.Value,
    pad: Sequence[ir.Value],
    mode: str = "constant",
    value: float | None = None,
) -> ir.Value:
    """pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor"""
    builder = current_builder()

    rank = len(self.shape)
    paddings = _process_padding(pad, rank)
    const_value = (
        builder.call("Constant", value=ir.tensor(value, dtype=ir.DataType(self.dtype)))[
            0
        ]
        if value is not None
        else None
    )
    onnx_mode = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }[mode]

    return builder.call("Pad", self, paddings, const_value, mode=onnx_mode)[0]


@onnx_impl(aten.reflection_pad1d.default)
def aten_reflection_pad1d(self: ir.Value, padding: Sequence[ir.Value]) -> ir.Value:
    """reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor"""
    builder = current_builder()

    # assert len(padding) == 2
    # Input of padding argument should be [x,y], need change to onnx format [0, x, 0, y]
    rank = len(self.shape)
    paddings = _process_padding(padding, rank)
    return builder.call("Pad", self, paddings, mode="reflect")[0]


@onnx_impl(aten.reflection_pad2d.default)
def aten_reflection_pad2d(self: ir.Value, padding: Sequence[ir.Value]) -> ir.Value:
    """reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor"""
    builder = current_builder()
    rank = len(self.shape)
    paddings = _process_padding(padding, rank)
    return builder.call("Pad", self, paddings, mode="reflect")[0]


@onnx_impl(aten.reflection_pad3d.default)
def aten_reflection_pad3d(self: ir.Value, padding: Sequence[ir.Value]) -> ir.Value:
    """reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor"""
    builder = current_builder()
    rank = len(self.shape)
    paddings = _process_padding(padding, rank)
    return builder.call("Pad", self, paddings, mode="reflect")[0]


@onnx_impl(aten.relu.default)
def aten_relu(self: ir.Value) -> ir.Value:
    """relu(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Relu", self)[0]


@onnx_impl(aten.relu6.default)
def aten_relu6(self: ir.Value) -> ir.Value:
    """relu6(Tensor self) -> Tensor"""
    builder = current_builder()

    zero = builder.call("CastLike", builder.constant(0, dtype=ir.DataType.INT64), self)[
        0
    ]
    six = builder.call("CastLike", builder.constant(6, dtype=ir.DataType.INT64), self)[
        0
    ]
    return builder.call("Clip", self, zero, six)[0]


@onnx_impl(aten.replication_pad1d.default)
def aten_replication_pad1d(self: ir.Value, padding: Sequence[ir.Value]) -> ir.Value:
    """replication_pad1d(Tensor self, SymInt[2] padding) -> Tensor"""
    builder = current_builder()

    rank = len(self.shape)
    paddings = _process_padding(padding, rank)
    return builder.call("Pad", self, paddings, mode="edge")[0]


@onnx_impl(aten.replication_pad2d.default)
def aten_replication_pad2d(self: ir.Value, padding: Sequence[ir.Value]) -> ir.Value:
    """replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor"""
    builder = current_builder()

    rank = len(self.shape)
    paddings = _process_padding(padding, rank)
    return builder.call("Pad", self, paddings, mode="edge")[0]


@onnx_impl(aten.replication_pad3d.default)
def aten_replication_pad3d(self: ir.Value, padding: Sequence[ir.Value]) -> ir.Value:
    """replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor"""
    builder = current_builder()

    rank = len(self.shape)
    paddings = _process_padding(padding, rank)
    return builder.call("Pad", self, paddings, mode="edge")[0]


def _causal_attention_mask(query: ir.Value, key: ir.Value) -> ir.Value:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float("inf"))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """
    builder = current_builder()
    q_shape = builder.call("Shape", query)[0]
    k_shape = builder.call("Shape", key)[0]

    target_length = builder.call(
        "Slice",
        q_shape,
        builder.constant([-2], dtype=ir.DataType.INT64),
        builder.constant([-1], dtype=ir.DataType.INT64),
    )[0]
    source_length = builder.call(
        "Slice",
        k_shape,
        builder.constant([-2], dtype=ir.DataType.INT64),
        builder.constant([-1], dtype=ir.DataType.INT64),
    )[0]
    # attn_mask = torch.ones(L, S) := {
    size = builder.call("Concat", target_length, source_length, axis=0)[0]
    attn_mask = builder.call(
        "Expand", builder.constant(1.0, dtype=ir.DataType.FLOAT), size
    )[0]
    # }
    attn_mask = builder.call("Trilu", attn_mask, upper=0)[0]
    # The causal mask has 0s in the lower triangle and the lowest representable value
    # of the query dtype in the upper triangle. The lowest finite value is used instead
    # of -inf to stay consistent with the boolean mask path and to avoid producing NaN
    # in the subsequent Softmax when a row is fully masked.
    # The constants are built in the query dtype directly rather than in float32 and
    # cast down, because casting float32's lowest value to float16 overflows to -inf.
    attn_mask = builder.call(
        "Where",
        builder.call(
            "Equal", attn_mask, builder.constant(0.0, dtype=ir.DataType.FLOAT)
        )[0],
        builder.call("Constant", value=ir.tensor(query.dtype.min, dtype=query.dtype))[
            0
        ],
        builder.call("Constant", value=ir.tensor(0.0, dtype=query.dtype))[0],
    )[0]
    return attn_mask


@onnx_impl(aten.scaled_dot_product_attention.default, opset_introduced=18)
def aten_scaled_dot_product_attention(
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    attn_mask: ir.Value | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> ir.Value:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Equivalent to the PyTorch code::
        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = (
            torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            if is_causal
            else attn_mask
        )
        attn_mask = (
            attn_mask.masked_fill(not attn_mask, -float("inf"))
            if attn_mask.dtype == torch.bool
            else attn_mask
        )
        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1
        )
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    builder = current_builder()
    # Optional inputs are handled during native construction.
    if is_causal and attn_mask is not None:
        raise ValueError(
            "is_causal and attn_mask cannot be set at the same time. "
            "Set is_causal=False or omit attn_mask."
        )

    if not (len(query.shape) == 4 and len(key.shape) == 4 and len(value.shape) == 4):
        raise ValueError(
            "Native opset18 attention requires 4D query, key, and value. "
            f"Got shapes {query.shape}, {key.shape}, and {value.shape}. "
            "Reshape inputs to [batch, heads, sequence, channels]."
        )

    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if scale is None:
        scale = _attention_scale(query)
    scale = builder.call("CastLike", scale, query)[0]

    if is_causal:
        attn_mask = _causal_attention_mask(query, key)

    if enable_gqa:
        if not isinstance(query.shape[1], int) or not isinstance(key.shape[1], int):
            raise ValueError(
                "Grouped-query attention needs known query and key head counts. "
                "Keep dimension 1 static when exporting with enable_gqa=True."
            )
        if query.shape[1] != key.shape[1]:
            key, value = _attention_repeat_kv_for_group_query(query, key, value)

    return _attention(query, key, value, attn_mask, scale, dropout_p)


@onnx_impl(aten._scaled_dot_product_flash_attention_for_cpu.default)
def aten__scaled_dot_product_flash_attention_for_cpu(
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_mask: ir.Value | None = None,
    scale: float | None = None,
) -> tuple[ir.Value, ir.Value]:
    """_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)"""
    builder = current_builder()
    result = aten_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    query_shape = builder.call("Shape", query)[0]
    query_first_dims = builder.call("Slice", query_shape, [0], [1])[0]
    query_second_dims = builder.call("Slice", query_shape, [1], [2])[0]
    num_heads = builder.call("Slice", query_shape, [-2], [-1])[0]
    logsumexp_dim = builder.call(
        "Cast",
        builder.call(
            "Mul",
            builder.call(
                "Ceil",
                builder.call(
                    "Div",
                    builder.call("Cast", query_second_dims, to=ir.DataType.FLOAT)[0],
                    32.0,
                )[0],
            )[0],
            32.0,
        )[0],
        to=ir.DataType.INT64,
    )[0]
    logsum_exp = builder.call(
        "Expand",
        0.0,
        builder.call("Concat", query_first_dims, num_heads, logsumexp_dim, axis=0)[0],
    )[0]
    return result, logsum_exp


def _attention(query, key, value, attn_mask, scale, dropout_p):
    builder = current_builder()
    # Swap the last two axes of key.
    key_shape = builder.call("Shape", key)[0]
    key_last_dim = builder.call("Slice", key_shape, [-1], [_INT64_MAX])[0]
    key_second_last_dim = builder.call("Slice", key_shape, [-2], [-1])[0]
    key_first_dims = builder.call("Slice", key_shape, [_INT64_MIN], [-2])[0]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = builder.call(
        "Concat", [-1], key_second_last_dim, key_last_dim, axis=0
    )[0]
    key_squeezed = builder.call("Reshape", key, key_squeezed_shape)[0]
    key_squeezed_transposed = builder.call("Transpose", key_squeezed, perm=[0, 2, 1])[0]
    key_transposed_shape = builder.call(
        "Concat", key_first_dims, key_last_dim, key_second_last_dim, axis=0
    )[0]
    key_transposed = builder.call(
        "Reshape", key_squeezed_transposed, key_transposed_shape
    )[0]
    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math.
    query_scaled = builder.call("Mul", query, builder.call("Sqrt", scale)[0])[0]
    key_scaled = builder.call(
        "Mul",
        key_transposed,
        builder.call("CastLike", builder.call("Sqrt", scale)[0], key_transposed)[0],
    )[0]
    logits = builder.call("MatMul", query_scaled, key_scaled)[0]
    boolean_mask = attn_mask is not None and attn_mask.dtype == ir.DataType.BOOL
    zero = builder.constant(0.0, dtype=query.dtype)
    if attn_mask is not None:
        mask = attn_mask
        if boolean_mask:
            # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf')).
            mask = builder.call(
                "Where",
                mask,
                zero,
                builder.constant(query.dtype.min, dtype=query.dtype),
            )[0]
        logits = builder.call("Add", logits, mask)[0]
    attn_weight = builder.call("Softmax", logits, axis=-1)[0]
    # When using scaled dot product attention with a boolean mask, the softmax operation might return NaN values
    # due to the presence of -inf in an entire row (padding tokens), resulting in 0/0 (NaN) in the softmax output.
    # This is because there's no safe/masked softmax imp in ONNX, so we need to handle NaN values explicitly to match
    # the behavior of PyTorch with boolean masks.
    # Reference: https://github.com/pytorch/pytorch/issues/103749
    attn_weight = builder.call(
        "Where", builder.call("IsNaN", attn_weight)[0], zero, attn_weight
    )[0]
    if boolean_mask:
        attn_weight = builder.call("Where", attn_mask, attn_weight, zero)[0]
    if dropout_p == 1.0:
        attn_weight = builder.call(
            "Expand", zero, builder.call("Shape", attn_weight)[0]
        )[0]
    elif dropout_p != 0:
        attn_weight, _ = builder.call(
            "Dropout", attn_weight, dropout_p, True, _num_outputs=2
        )
    return builder.call("MatMul", attn_weight, value)[0]


@onnx_impl(aten.silu.default)
def aten_silu(self: ir.Value) -> ir.Value:
    """silu(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Mul", self, builder.call("Sigmoid", self)[0])[0]


@onnx_impl(aten.softplus.default)
def aten_softplus(
    self: ir.Value, beta: float = 1.0, threshold: float = 20.0
) -> ir.Value:
    """softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor"""
    builder = current_builder()

    self_scaled = builder.call("Mul", self, beta)[0]
    softplus = builder.call("Div", builder.call("Softplus", self_scaled)[0], beta)[0]
    return builder.call(
        "Where", builder.call("Greater", self_scaled, threshold)[0], self, softplus
    )[0]


def _get_upsample_align_corners_mode(align_corners: bool) -> str:
    return "align_corners" if align_corners else "half_pixel"


def _aten_upsample_output_size(
    self: ir.Value,
    output_size: ir.Value,
    mode: str,
    coordinate_transformation_mode: str,
    antialias: int = 0,
    cubic_coeff_a: float = -0.75,
    exclude_outside: int = 0,
) -> ir.Value:
    builder = current_builder()
    batch_and_channel = builder.call("Shape", self, end=2, start=0)[0]
    # When output_size is passed in as a list of integers, the torch.onnx
    # graph builder when handling op.Concat may fail
    # to determine the output type. We cast it to INT64 to ensure the output
    output_size = common_ops.merge_dims(output_size)
    # Append the batch and channel dimensions to the requested output size
    output_size = builder.call("Concat", batch_and_channel, output_size, axis=0)[0]
    return builder.call(
        "Resize",
        self,
        None,
        None,
        output_size,
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
        cubic_coeff_a=cubic_coeff_a,
        nearest_mode="floor",
        antialias=antialias,
        exclude_outside=exclude_outside,
    )[0]


def _aten_upsample_scales(
    self: ir.Value,
    scale_factors: Sequence[float],
    mode: str,
    coordinate_transformation_mode: str,
    antialias: int = 0,
    cubic_coeff_a: float = -0.75,
    exclude_outside: int = 0,
) -> ir.Value:
    builder = current_builder()
    # For 2D the format is [1.0, 1.0, scale_h, scale_w]; include every spatial axis.
    return builder.call(
        "Resize",
        self,
        None,
        builder.constant(
            [1.0] * (len(self.shape) - len(scale_factors)) + list(scale_factors),
            dtype=ir.DataType.FLOAT,
        ),
        None,
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
        cubic_coeff_a=cubic_coeff_a,
        nearest_mode="floor",
        antialias=antialias,
        exclude_outside=exclude_outside,
    )[0]


@onnx_impl(aten.upsample_bicubic2d.default)
def aten_upsample_bicubic2d(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: Based on experimentation, scales_h and scales_w are always ignored in PyTorch,
    # unless when align_corners is True, in which case we do not know what is going on.
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="cubic",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@onnx_impl(aten._upsample_bicubic2d_aa.default)
def aten__upsample_bicubic2d_aa(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """_upsample_bicubic2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: Based on experimentation, scales_h and scales_w are always ignored in PyTorch,
    # unless when align_corners is True, in which case we do not know what is going on.
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    # PyTorch uses cubic_coeff_a=-0.5 (Keys interpolation, PIL-compatible) when
    # antialias=True, as opposed to -0.75 (OpenCV-compatible) for the non-antialias case.
    # exclude_outside=1 matches PyTorch's antialias kernel, which drops out-of-bounds
    # samples from the filter window and renormalizes the remaining weights so they
    # sum to 1. Without it the ONNX Resize keeps phantom out-of-bounds weight in the
    # denominator and produces values that differ from eager near the boundary.
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="cubic",
        coordinate_transformation_mode=coordinate_transformation_mode,
        antialias=1,
        cubic_coeff_a=-0.5,
        exclude_outside=1,
    )


@onnx_impl(aten.upsample_bicubic2d.vec)
def aten_upsample_bicubic2d_vec(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> ir.Value:
    """upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        result = _aten_upsample_scales(
            self,
            scale_factors,
            mode="cubic",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    else:
        result = _aten_upsample_output_size(
            self,
            output_size,
            mode="cubic",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )

    return result


@onnx_impl(aten.upsample_bilinear2d.default)
def aten_upsample_bilinear2d(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: Based on experimentation, scales_h and scales_w are always ignored in PyTorch,
    # unless when align_corners is True, in which case we do not know what is going on.
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        coordinate_transformation_mode=coordinate_transformation_mode,
        mode="linear",
    )


@onnx_impl(aten._upsample_bilinear2d_aa.default)
def aten__upsample_bilinear2d_aa(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """_upsample_bilinear2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: Based on experimentation, scales_h and scales_w are always ignored in PyTorch,
    # unless when align_corners is True, in which case we do not know what is going on.
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    # exclude_outside=1 matches PyTorch's antialias kernel, which drops out-of-bounds
    # samples from the filter window and renormalizes the remaining weights so they
    # sum to 1. Without it the ONNX Resize keeps phantom out-of-bounds weight in the
    # denominator and produces values that differ from eager near the boundary.
    return _aten_upsample_output_size(
        self,
        output_size,
        coordinate_transformation_mode=coordinate_transformation_mode,
        mode="linear",
        antialias=1,
        exclude_outside=1,
    )


@onnx_impl(aten.upsample_bilinear2d.vec)
def aten_upsample_bilinear2d_vec(
    self: ir.Value,
    output_size: ir.Value | None,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> ir.Value:
    """upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        result = _aten_upsample_scales(
            self,
            scale_factors,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    else:
        if output_size is None:
            raise ValueError(
                "aten::upsample_bilinear2d.vec received neither output_size nor "
                "scale_factors. Supply one of them to determine the output shape."
            )
        result = _aten_upsample_output_size(
            self,
            output_size,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )

    return result


@onnx_impl(aten.upsample_linear1d.default)
def aten_upsample_linear1d(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scales: float | None = None,
) -> ir.Value:
    """upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None) -> Tensor"""
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    # scales is ignored in PyTorch
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="linear",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@onnx_impl(aten.upsample_linear1d.vec)
def aten_upsample_linear1d_vec(
    self: ir.Value,
    output_size: ir.Value | None,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> ir.Value:
    """upsample_linear1d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        return _aten_upsample_scales(
            self,
            scale_factors,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    if output_size is None:
        raise ValueError(
            "aten::upsample_linear1d.vec received neither output_size nor "
            "scale_factors. Supply one of them to determine the output shape."
        )
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="linear",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@onnx_impl(aten.upsample_nearest1d.default)
def aten_upsample_nearest1d(
    self: ir.Value, output_size: ir.Value, scales: float | None = None
) -> ir.Value:
    """upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor"""
    if scales is not None:
        return _aten_upsample_scales(self, [scales], "nearest", "asymmetric")
    else:
        return _aten_upsample_output_size(self, output_size, "nearest", "asymmetric")


@onnx_impl(
    (
        aten.upsample_nearest1d.vec,
        aten.upsample_nearest2d.vec,
        aten.upsample_nearest3d.vec,
    )
)
def aten_upsample_nearestnd_vec(
    input: ir.Value,
    output_size: ir.Value | None = None,
    scale_factors: Sequence[float] | None = None,
) -> ir.Value:
    """upsample_nearest3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor"""

    if scale_factors is not None:
        return _aten_upsample_scales(input, scale_factors, "nearest", "asymmetric")
    else:
        if output_size is None:
            raise ValueError(
                "aten::upsample_nearestnd.vec received neither output_size nor "
                "scale_factors. Supply one of them to determine the output shape."
            )
        return _aten_upsample_output_size(input, output_size, "nearest", "asymmetric")


@onnx_impl(aten.upsample_nearest2d.default)
def aten_upsample_nearest2d(
    self: ir.Value,
    output_size: ir.Value,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor"""

    if scales_h is not None and scales_w is not None:
        return _aten_upsample_scales(
            self,
            [scales_h, scales_w],
            "nearest",
            "asymmetric",
        )
    else:
        return _aten_upsample_output_size(self, output_size, "nearest", "asymmetric")


@onnx_impl(aten.upsample_nearest3d.default)
def aten_upsample_nearest3d(
    self: ir.Value,
    output_size: ir.Value,
    scales_d: float | None = None,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""

    if scales_d is not None and scales_h is not None and scales_w is not None:
        return _aten_upsample_scales(
            self,
            [scales_d, scales_h, scales_w],
            "nearest",
            "asymmetric",
        )
    else:
        return _aten_upsample_output_size(self, output_size, "nearest", "asymmetric")


@onnx_impl(aten.upsample_trilinear3d.default)
def aten_upsample_trilinear3d(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scales_d: float | None = None,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> ir.Value:
    """upsample_trilinear3d(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""

    del scales_d
    del scales_h
    del scales_w

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="linear",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@onnx_impl(aten.upsample_trilinear3d.vec)
def aten_upsample_trilinear3d_vec(
    self: ir.Value,
    output_size: ir.Value,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> ir.Value:
    """upsample_trilinear3d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        result = _aten_upsample_scales(
            self,
            scale_factors,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    else:
        result = _aten_upsample_output_size(
            self,
            output_size,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    return result
