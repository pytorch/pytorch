# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR normalization translations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from torch.onnx._internal.exporter._ir_builder import TranslationContext


aten = torch.ops.aten
_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808


def _conv(
    op: TranslationContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dimensions,
    *,
    complex,
):
    builder = op.builder
    # Attributes need to be manipulated in Python to match ONNX's conv1d/2d/3d.
    padding = list(padding) if isinstance(padding, Sequence) else [padding] * dimensions
    dilation = (
        list(dilation) if isinstance(dilation, Sequence) else [dilation] * dimensions
    )
    stride = list(stride) if isinstance(stride, Sequence) else [stride] * dimensions
    if bias is None:
        weight_dim_0 = op.Shape(weight, start=0, end=1)
        bias_shape = op.Concat(weight_dim_0, [2], axis=0) if complex else weight_dim_0
        bias = op.Expand(builder.constant(0.0, dtype=input.dtype), bias_shape)
    impl = _aten_convolution_complex_onnx if complex else _aten_convolution_onnx
    return impl(
        op,
        input,
        weight,
        bias,
        transposed=False,
        strides=stride,
        pads=[*padding, *padding],
        dilations=dilation,
        groups=groups,
    )


@onnx_impl(aten.conv1d.default)
def aten_conv1d(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
) -> ir.Value:
    """conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor"""

    return _conv(
        op, input, weight, bias, stride, padding, dilation, groups, 1, complex=False
    )


@onnx_impl(aten.conv1d.default, complex=True)
def aten_conv1d_complex(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
) -> ir.Value:
    """conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor"""

    return _conv(
        op, input, weight, bias, stride, padding, dilation, groups, 1, complex=True
    )


@onnx_impl(aten.conv2d.default)
def aten_conv2d(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> ir.Value:
    """conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor"""

    return _conv(
        op, input, weight, bias, stride, padding, dilation, groups, 2, complex=False
    )


@onnx_impl(aten.conv2d.default, complex=True)
def aten_conv2d_complex(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> ir.Value:
    """conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor"""

    return _conv(
        op, input, weight, bias, stride, padding, dilation, groups, 2, complex=True
    )


@onnx_impl(aten.conv3d.default)
def aten_conv3d(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
) -> ir.Value:
    """conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor"""

    return _conv(
        op, input, weight, bias, stride, padding, dilation, groups, 3, complex=False
    )


@onnx_impl(aten.conv3d.default, complex=True)
def aten_conv3d_complex(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
) -> ir.Value:
    """conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor"""

    return _conv(
        op, input, weight, bias, stride, padding, dilation, groups, 3, complex=True
    )


@onnx_impl(aten.convolution.default)
def aten_convolution(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    transposed: bool = False,
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> ir.Value:
    """convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor"""

    rank = len(input.shape)

    image_d = rank - 2

    # NOTE: We assume the sequence padding/dilation/stride
    # from ATen op can only be either len == 1 or
    # len == rank.

    if not isinstance(padding, Sequence):
        padding = [padding] * image_d
    elif len(padding) == 1:
        padding = [padding[0]] * image_d
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = [dilation] * image_d
    elif len(dilation) == 1:
        dilation = [dilation[0]] * image_d
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = [stride] * image_d
    elif len(stride) == 1:
        stride = [stride[0]] * image_d
    strides = list(stride)

    result = _aten_convolution_onnx(
        op,
        input,
        weight,
        bias,
        transposed,
        strides=strides,
        pads=pads,
        dilations=dilations,
        output_padding=output_padding,
        groups=groups,
    )

    return result


def _conv_kernel_shape(
    op: TranslationContext, weight: ir.Value, complex: bool = False
) -> Sequence[int] | None:
    """Return the spatial (kernel) shape of the convolution weight if statically known.

    The kernel_shape attribute of ONNX Conv/ConvTranspose corresponds to the spatial
    dimensions of the weight tensor, i.e. all dimensions except the leading two
    (output and input channels). When ``complex`` is True, the weight has a trailing
    dimension of size 2 (real/imaginary parts) which is also excluded. Returns None
    when the spatial shape is not fully static, in which case the attribute is omitted
    and inferred by ONNX.
    """
    shape = weight.shape
    if shape is None:
        return None
    # Exclude the leading output/input channel dims, and for complex weights the
    # trailing real/imag dim as well.
    kernel_shape = shape[2:-1] if complex else shape[2:]
    if len(kernel_shape) == 0:
        return None
    if any(not isinstance(dim, int) for dim in kernel_shape):
        return None
    return list(kernel_shape)


def _aten_convolution_onnx(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value,
    transposed: bool,
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> ir.Value:
    """ConvXd with attributes pre-computed to fit the ONNX spec."""
    builder = op.builder

    # Native construction branches on the Python transposed flag, not a graph input.

    no_batch = len(input.shape) != len(weight.shape)

    if no_batch:
        input = op.Unsqueeze(input, builder.constant([0], dtype=ir.DataType.INT64))

    # kernel_shape is the spatial shape of the weight tensor (excluding the
    # output-channel and input-channel dimensions). It is optional in the ONNX
    # spec but is set explicitly here to match the ONNX Conv specification.
    kernel_shape = _conv_kernel_shape(op, weight)

    if transposed:
        result = op.ConvTranspose(
            input,
            weight,
            bias,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            output_padding=output_padding,
            kernel_shape=kernel_shape,
        )
    else:
        result = op.Conv(
            input,
            weight,
            bias,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            kernel_shape=kernel_shape,
        )

    if no_batch:
        result = op.Squeeze(result, builder.constant([0], dtype=ir.DataType.INT64))

    return result


def _aten_convolution_complex_onnx(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value,
    transposed: bool,
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> ir.Value:
    """ConvXd on complex inputs with attributes pre-computed to fit the ONNX spec."""
    builder = op.builder

    # Native construction branches on the Python transposed flag, not a graph input.

    no_batch = len(input.shape) != len(weight.shape)

    if no_batch:
        input = op.Unsqueeze(input, builder.constant([0], dtype=ir.DataType.INT64))

    input_real = op.Gather(input, 0, axis=-1)
    input_imag = op.Gather(input, 1, axis=-1)
    weight_real = op.Gather(weight, 0, axis=-1)
    weight_imag = op.Gather(weight, 1, axis=-1)
    bias_real = op.Gather(bias, 0, axis=-1)
    bias_imag = op.Gather(bias, 1, axis=-1)
    bias_zero = op.Expand(
        op.CastLike(0.0, weight),
        op.Shape(bias_real),
    )

    # The complex weight has a trailing dimension of size 2 (real/imaginary parts),
    # so the spatial (kernel) shape excludes both leading channel dimensions and the
    # trailing complex dimension.
    kernel_shape = _conv_kernel_shape(op, weight, complex=True)

    if transposed:
        result_real = op.Sub(
            op.ConvTranspose(
                input_real,
                weight_real,
                bias_real,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            ),
            op.ConvTranspose(
                input_imag,
                weight_imag,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            ),
        )
        result_imag = op.Add(
            op.ConvTranspose(
                input_real,
                weight_imag,
                bias_imag,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            ),
            op.ConvTranspose(
                input_imag,
                weight_real,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            ),
        )
    else:
        result_real = op.Sub(
            op.Conv(
                input_real,
                weight_real,
                bias_real,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            ),
            op.Conv(
                input_imag,
                weight_imag,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            ),
        )

        result_imag = op.Add(
            op.Conv(
                input_real,
                weight_imag,
                bias_imag,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            ),
            op.Conv(
                input_imag,
                weight_real,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            ),
        )

    result = op.Concat(
        op.Unsqueeze(result_real, [-1]),
        op.Unsqueeze(result_imag, [-1]),
        axis=-1,
    )

    if no_batch:
        result = op.Squeeze(result, builder.constant([0], dtype=ir.DataType.INT64))

    return result


@onnx_impl(aten.instance_norm.default)
def aten_instance_norm(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    running_mean: ir.Value | None = None,
    running_var: ir.Value | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
    cudnn_enabled: bool = False,
) -> ir.Value:
    """instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor"""
    builder = op.builder
    del cudnn_enabled  # unused
    if weight is None:  # Set to 1.0 as default
        weight = op.CastLike(
            op.Expand(
                builder.constant([1.0], dtype=input.dtype),
                op.Shape(input, start=1, end=2),
            ),
            input,
        )

    if bias is None:  # Set to 0.0 as default
        bias = op.CastLike(
            op.Expand(
                builder.constant([0.0], dtype=input.dtype),
                op.Shape(input, start=1, end=2),
            ),
            input,
        )

    # If `use_input_stats` is set to True, ignore 'running_mean' and 'running_var' and
    # compute using input statistics.
    # Otherwise, compute using the running statistics.
    if use_input_stats:
        return op.InstanceNormalization(input, weight, bias, epsilon=eps)

    if not (running_mean is not None and running_var is not None):
        raise ValueError(
            "running_mean and running_var must be provided when use_input_stats "
            "is False. Supply both statistics or set use_input_stats=True."
        )

    batch_size = op.Shape(input, start=0, end=1)
    bn_input = op.Reshape(
        input,
        op.Concat(
            builder.constant([1, -1], dtype=ir.DataType.INT64),
            op.Shape(input, start=2),
            axis=0,
        ),
    )
    weight = op.Tile(weight, batch_size)
    bias = op.Tile(bias, batch_size)
    running_mean = op.Tile(running_mean, batch_size)
    running_var = op.Tile(running_var, batch_size)

    norm = op.BatchNormalization(
        bn_input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=1.0 - momentum,
        training_mode=False,
        outputs=1,
    )[0]
    return op.Reshape(norm, op.Shape(input), allowzero=True)


@onnx_impl(aten.layer_norm.default)
def aten_layer_norm(
    op: TranslationContext,
    input: ir.Value,
    normalized_shape: Sequence[int | ir.Value],
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
    cudnn_enable: bool = True,
) -> ir.Value:
    """layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"""

    # Use Python metadata to obtain start_axis.
    start_axis = -len(normalized_shape)

    if weight is None:
        one = op.Constant(value=ir.tensor(1, dtype=input.dtype))
        weight = op.Expand(one, op.Shape(input, start=start_axis))

    result, _, _ = op.LayerNormalization(
        input,
        weight,
        bias,
        axis=start_axis,
        epsilon=eps,
        outputs=3,
    )
    return result


@onnx_impl(aten._native_batch_norm_legit_no_training.default)
def aten__native_batch_norm_no_training(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    running_mean: ir.Value | None = None,
    running_var: ir.Value | None = None,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""

    return aten_native_batch_norm(
        op, input, weight, bias, running_mean, running_var, False, momentum, eps
    )


@onnx_impl(aten._native_batch_norm_legit.no_stats)
def aten__native_batch_norm_no_stats(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""

    return aten_native_batch_norm(
        op, input, weight, bias, None, None, training, momentum, eps
    )


@onnx_impl((aten.native_batch_norm.default, aten._native_batch_norm_legit.default))
def aten_native_batch_norm(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    running_mean: ir.Value | None = None,
    running_var: ir.Value | None = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""

    return aten__native_batch_norm_legit_functional(
        op, input, weight, bias, running_mean, running_var, training, momentum, eps
    )[:3]


def _aten_native_batch_norm_training_onnx(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value,
    running_mean: ir.Value,
    running_var: ir.Value,
    axes: ir.Value,
    momentum: float,
    eps: float,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
    """Batch normalization training mode.

    NOTE: momentum in PyTorch is 1.0-momentum in ONNX.
    When calling this function be sure to pass 1.0-momentum when momentum is obtained from PyTorch.
    """
    norm, running_mean, _ = op.BatchNormalization(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=momentum,
        training_mode=True,
        outputs=3,
    )
    # Compute mean and rstd
    # Mean, var, and rstd computation and results are expected to be
    # in higher precision when inputs are float16.
    upcast_input = op.Cast(
        input,
        to=ir.DataType.DOUBLE
        if input.dtype == ir.DataType.DOUBLE
        else ir.DataType.FLOAT,
    )
    mean = op.ReduceMean(upcast_input, axes)
    input_sub_mean = op.Sub(upcast_input, mean)
    sqr = op.Mul(input_sub_mean, input_sub_mean)
    var = op.ReduceMean(sqr, axes, keepdims=False)
    rstd = op.Div(1.0, op.Sqrt(op.Add(var, eps)))
    # Get mean again with size = [1, C]
    mean = op.ReduceMean(upcast_input, axes, keepdims=False)

    # Compute the running var the PyTorch way
    # https://github.com/pytorch/pytorch/blob/5cc511f72fe073bbd8c10d796d72dce67f5cd5c4/torch/_decomp/decompositions.py#L1646

    n = op.Cast(
        op.Div(
            op.Size(input),
            op.Gather(op.Shape(input), 1, axis=0),
        ),
        to=upcast_input.dtype,
    )
    unbiased_var = op.Mul(var, op.Div(n, op.Sub(n, 1.0)))

    # NOTE: momentum in ONNX is 1.0-momentum in PyTorch
    new_running_var = op.Add(
        op.CastLike(
            op.Mul(1.0 - momentum, unbiased_var),
            running_var,
        ),
        op.Mul(momentum, running_var),
    )

    return norm, mean, rstd, running_mean, new_running_var


def _aten_native_batch_norm_inference_onnx(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    bias: ir.Value,
    running_mean: ir.Value,
    running_var: ir.Value,
    momentum: float,
    eps: float,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
    """Batch normalization inference mode.

    NOTE: momentum in PyTorch is 1.0-momentum in ONNX.
    When calling this function be sure to pass 1.0-momentum when momentum is obtained from PyTorch.
    """
    norm = op.BatchNormalization(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=momentum,
        training_mode=False,
        outputs=1,
    )[0]
    # CUDA and CPU gives different shapes:
    # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1451-L1457
    # We use CUDA's output here
    invstd = op.Div(1.0, op.Sqrt(op.Add(running_var, eps)))
    # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1475
    running_mean_fp32 = op.Cast(running_mean, to=ir.DataType.FLOAT)
    invstd = op.Cast(invstd, to=ir.DataType.FLOAT)
    return (
        norm,
        running_mean_fp32,
        invstd,
        op.Identity(running_mean),
        op.Identity(running_var),
    )


@onnx_impl(aten._native_batch_norm_legit_functional.default)
def aten__native_batch_norm_legit_functional(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    running_mean: ir.Value | None = None,
    running_var: ir.Value | None = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
    builder = op.builder
    if weight is None:  # Set to 1.0 as default
        weight = op.Expand(
            builder.constant([1.0], dtype=input.dtype),
            op.Shape(input, start=1, end=2),
        )

    if bias is None:  # Set to 0.0 as default
        bias = op.Expand(
            builder.constant([0.0], dtype=input.dtype),
            op.Shape(input, start=1, end=2),
        )

    axes = list(range(len(input.shape)))
    axes.pop(1)
    axes = builder.constant(axes, dtype=ir.DataType.INT64)
    if running_mean is None:  # Using input mean
        running_mean = op.ReduceMean(input, axes, keepdims=False)

    if running_var is None:  # Using input var
        mean = op.ReduceMean(input, axes)
        input_sub_mean = op.Sub(input, mean)
        sqr_input_sub_mean = op.Mul(input_sub_mean, input_sub_mean)
        running_var = op.ReduceMean(sqr_input_sub_mean, axes, keepdims=False)

    # Preserve training rather than forcing inference for older ORT kernels.
    # We have to split to two private functions, because BatchNormalization returns
    # three outputs when training_mode=True and one when it is False.
    if training:
        (
            norm,
            input_mean,
            input_rstd,
            running_mean,
            running_var,
        ) = _aten_native_batch_norm_training_onnx(
            op,
            input,
            weight,
            bias,
            running_mean,
            running_var,
            axes,
            momentum=1.0 - momentum,
            eps=eps,
        )
    else:
        (
            norm,
            input_mean,
            input_rstd,
            running_mean,
            running_var,
        ) = _aten_native_batch_norm_inference_onnx(
            op,
            input,
            weight,
            bias,
            running_mean,
            running_var,
            momentum=1.0 - momentum,
            eps=eps,
        )

    return norm, input_mean, input_rstd, running_mean, running_var


@onnx_impl(aten.native_group_norm.default)
def aten_native_group_norm(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    N: ir.Value | None = None,
    C: ir.Value | None = None,
    HxW: ir.Value | None = None,
    group: int = 1,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)"""
    builder = op.builder

    # Actually we don't need N,C,HxW value because the input tensor has that information
    if weight is None:  # Set to 1.0 as default, the shape is Channel size
        weight = op.Expand(
            builder.constant([1.0], dtype=ir.DataType.FLOAT),
            op.Shape(input, start=1, end=2),
        )

    if bias is None:  # Set to 0.0 as default, the shape is Channel size
        bias = op.Expand(
            builder.constant([0.0], dtype=ir.DataType.FLOAT),
            op.Shape(input, start=1, end=2),
        )

    # Because onnx.GroupNorm() need size=group for weight and bias
    # But the torch's aten function's input need size=channel, the size mismatched
    # So we have to use onnx.InstanceNorm() to simulate
    # This implementation should be simplified after opset 21
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    # Create weight_instance_norm and bias_instance_norm, copied from Torch ONNX converter
    group_tensor = op.Reshape(group, neg_1)
    # 0 in the shape list keeps dimension value unchanged, for InstanceNorm need [0,group,-1]
    shape_input = op.Concat(
        builder.constant([0], dtype=ir.DataType.INT64),
        group_tensor,
        neg_1,
        axis=0,
    )
    input_reshaped = op.Reshape(input, shape_input)
    weight_inst_norm = op.Expand(op.CastLike(1.0, input), group_tensor)
    bias_inst_norm = op.Expand(op.CastLike(0.0, input), group_tensor)
    norm = op.InstanceNormalization(
        input_reshaped,
        weight_inst_norm,
        bias_inst_norm,
        epsilon=eps,
    )
    # Reshape back to input's shape
    norm = op.Reshape(norm, op.Shape(input), allowzero=True)
    # Using the input weight and bias to do affine
    # But need to unsqueeze to the target shape for broading cast easy
    input_rank = len(input.shape)
    axes_unsqueeze = op.Range(1, input_rank - 1, 1)
    weight_full_shape = op.Unsqueeze(weight, axes_unsqueeze)
    bias_full_shape = op.Unsqueeze(bias, axes_unsqueeze)
    weight_full_shape = op.CastLike(weight_full_shape, norm)
    norm_mul_weight = op.Mul(norm, weight_full_shape)
    bias_full_shape = op.CastLike(bias_full_shape, norm_mul_weight)
    norm_result = op.Add(norm_mul_weight, bias_full_shape)
    # Compute mean and rstd, but using Torch algorithm
    # The returned shape for mean and vstd should be [N, group, -1]
    N = op.Shape(input, start=0, end=1)
    shape_N_group_neg1 = op.Concat(N, group_tensor, neg_1, axis=0)
    input_N_group_neg1 = op.Reshape(input, shape_N_group_neg1)
    # The output size is [N, group], so dims = [2]
    axes = builder.constant([2], dtype=ir.DataType.INT64)
    # Get mean which size is [N, group, 1], for broadcasting
    mean = op.ReduceMean(input_N_group_neg1, axes)
    input_sub_mean = op.Sub(input_N_group_neg1, mean)
    sqr_input_sub_mean = op.Mul(input_sub_mean, input_sub_mean)
    # In Pytorch, vstd = 1/(sqrt(var + eps))
    var = op.ReduceMean(sqr_input_sub_mean, axes, keepdims=False)
    eps = op.Constant(value=ir.tensor(eps, dtype=input.dtype))
    one = op.Constant(value=ir.tensor(1.0, dtype=input.dtype))
    rstd = op.Div(one, op.Sqrt(op.Add(var, eps)))
    # Get the correct shape [N, group] for mean again
    mean = op.ReduceMean(input_N_group_neg1, axes, keepdims=False)
    return norm_result, mean, rstd


@onnx_impl(aten.native_layer_norm.default)
def aten_native_layer_norm(
    op: TranslationContext,
    input: ir.Value,
    normalized_shape: Sequence[int],
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"""
    builder = op.builder

    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    # The mean and standard-deviation are calculated over the last D dimensions,
    # where D is the dimension of normalized_shape. For example, if normalized_shape is
    # (3, 5) (a 2-dimensional shape), the mean and standard-deviation are computed
    # over the last 2 dimensions of the input (i.e. input.mean((-2, -1))).

    # Use Python to manipulate
    start_axis = -len(normalized_shape)

    if weight is None:
        one = builder.constant([1.0], dtype=ir.DataType.FLOAT)
        weight = op.Expand(one, op.Shape(input, start=start_axis))
        weight = op.CastLike(weight, input)

    result, mean, rdenominator = op.LayerNormalization(
        input,
        weight,
        bias,
        axis=start_axis,
        epsilon=eps,
        outputs=3,
    )

    return result, mean, rdenominator
