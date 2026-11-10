# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR normalization translations."""

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten
_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808


def _conv(
    input, weight, bias, stride, padding, dilation, groups, dimensions, *, complex
):
    builder = current_builder()
    # Attributes need to be manipulated in Python to match ONNX's conv1d/2d/3d.
    padding = list(padding) if isinstance(padding, Sequence) else [padding] * dimensions
    dilation = (
        list(dilation) if isinstance(dilation, Sequence) else [dilation] * dimensions
    )
    stride = list(stride) if isinstance(stride, Sequence) else [stride] * dimensions
    if bias is None:
        weight_dim_0 = builder.call("Shape", weight, start=0, end=1)[0]
        bias_shape = (
            builder.call("Concat", weight_dim_0, [2], axis=0)[0]
            if complex
            else weight_dim_0
        )
        bias = builder.call(
            "Expand", builder.constant(0.0, dtype=input.dtype), bias_shape
        )[0]
    impl = _aten_convolution_complex_onnx if complex else _aten_convolution_onnx
    return impl(
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
        input, weight, bias, stride, padding, dilation, groups, 1, complex=False
    )


@onnx_impl(aten.conv1d.default, complex=True)
def aten_conv1d_complex(
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
        input, weight, bias, stride, padding, dilation, groups, 1, complex=True
    )


@onnx_impl(aten.conv2d.default)
def aten_conv2d(
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
        input, weight, bias, stride, padding, dilation, groups, 2, complex=False
    )


@onnx_impl(aten.conv2d.default, complex=True)
def aten_conv2d_complex(
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
        input, weight, bias, stride, padding, dilation, groups, 2, complex=True
    )


@onnx_impl(aten.conv3d.default)
def aten_conv3d(
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
        input, weight, bias, stride, padding, dilation, groups, 3, complex=False
    )


@onnx_impl(aten.conv3d.default, complex=True)
def aten_conv3d_complex(
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
        input, weight, bias, stride, padding, dilation, groups, 3, complex=True
    )


@onnx_impl(aten.convolution.default)
def aten_convolution(
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


def _conv_kernel_shape(weight: ir.Value, complex: bool = False) -> Sequence[int] | None:
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
    builder = current_builder()

    # Native construction branches on the Python transposed flag, not a graph input.

    no_batch = len(input.shape) != len(weight.shape)

    if no_batch:
        input = builder.call(
            "Unsqueeze", input, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]

    # kernel_shape is the spatial shape of the weight tensor (excluding the
    # output-channel and input-channel dimensions). It is optional in the ONNX
    # spec but is set explicitly here to match the ONNX Conv specification.
    kernel_shape = _conv_kernel_shape(weight)

    if transposed:
        result = builder.call(
            "ConvTranspose",
            input,
            weight,
            bias,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            output_padding=output_padding,
            kernel_shape=kernel_shape,
        )[0]
    else:
        result = builder.call(
            "Conv",
            input,
            weight,
            bias,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            kernel_shape=kernel_shape,
        )[0]

    if no_batch:
        result = builder.call(
            "Squeeze", result, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]

    return result


def _aten_convolution_complex_onnx(
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
    builder = current_builder()

    # Native construction branches on the Python transposed flag, not a graph input.

    no_batch = len(input.shape) != len(weight.shape)

    if no_batch:
        input = builder.call(
            "Unsqueeze", input, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]

    input_real = builder.call("Gather", input, 0, axis=-1)[0]
    input_imag = builder.call("Gather", input, 1, axis=-1)[0]
    weight_real = builder.call("Gather", weight, 0, axis=-1)[0]
    weight_imag = builder.call("Gather", weight, 1, axis=-1)[0]
    bias_real = builder.call("Gather", bias, 0, axis=-1)[0]
    bias_imag = builder.call("Gather", bias, 1, axis=-1)[0]
    bias_zero = builder.call(
        "Expand",
        builder.call("CastLike", 0.0, weight)[0],
        builder.call("Shape", bias_real)[0],
    )[0]

    # The complex weight has a trailing dimension of size 2 (real/imaginary parts),
    # so the spatial (kernel) shape excludes both leading channel dimensions and the
    # trailing complex dimension.
    kernel_shape = _conv_kernel_shape(weight, complex=True)

    if transposed:
        result_real = builder.call(
            "Sub",
            builder.call(
                "ConvTranspose",
                input_real,
                weight_real,
                bias_real,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            )[0],
            builder.call(
                "ConvTranspose",
                input_imag,
                weight_imag,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            )[0],
        )[0]
        result_imag = builder.call(
            "Add",
            builder.call(
                "ConvTranspose",
                input_real,
                weight_imag,
                bias_imag,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            )[0],
            builder.call(
                "ConvTranspose",
                input_imag,
                weight_real,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                output_padding=output_padding,
                kernel_shape=kernel_shape,
            )[0],
        )[0]
    else:
        result_real = builder.call(
            "Sub",
            builder.call(
                "Conv",
                input_real,
                weight_real,
                bias_real,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            )[0],
            builder.call(
                "Conv",
                input_imag,
                weight_imag,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            )[0],
        )[0]

        result_imag = builder.call(
            "Add",
            builder.call(
                "Conv",
                input_real,
                weight_imag,
                bias_imag,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            )[0],
            builder.call(
                "Conv",
                input_imag,
                weight_real,
                bias_zero,
                strides=strides,
                pads=pads,
                group=groups,
                dilations=dilations,
                kernel_shape=kernel_shape,
            )[0],
        )[0]

    result = builder.call(
        "Concat",
        builder.call("Unsqueeze", result_real, [-1])[0],
        builder.call("Unsqueeze", result_imag, [-1])[0],
        axis=-1,
    )[0]

    if no_batch:
        result = builder.call(
            "Squeeze", result, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]

    return result


@onnx_impl(aten.instance_norm.default)
def aten_instance_norm(
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
    builder = current_builder()
    del cudnn_enabled  # unused
    if weight is None:  # Set to 1.0 as default
        weight = builder.call(
            "CastLike",
            builder.call(
                "Expand",
                builder.constant([1.0], dtype=input.dtype),
                builder.call("Shape", input, start=1, end=2)[0],
            )[0],
            input,
        )[0]

    if bias is None:  # Set to 0.0 as default
        bias = builder.call(
            "CastLike",
            builder.call(
                "Expand",
                builder.constant([0.0], dtype=input.dtype),
                builder.call("Shape", input, start=1, end=2)[0],
            )[0],
            input,
        )[0]

    # If `use_input_stats` is set to True, ignore 'running_mean' and 'running_var' and
    # compute using input statistics.
    # Otherwise, compute using the running statistics.
    if use_input_stats:
        return builder.call("InstanceNormalization", input, weight, bias, epsilon=eps)[
            0
        ]

    if not (running_mean is not None and running_var is not None):
        raise ValueError(
            "running_mean and running_var must be provided when use_input_stats "
            "is False. Supply both statistics or set use_input_stats=True."
        )

    batch_size = builder.call("Shape", input, start=0, end=1)[0]
    bn_input = builder.call(
        "Reshape",
        input,
        builder.call(
            "Concat",
            builder.constant([1, -1], dtype=ir.DataType.INT64),
            builder.call("Shape", input, start=2)[0],
            axis=0,
        )[0],
    )[0]
    weight = builder.call("Tile", weight, batch_size)[0]
    bias = builder.call("Tile", bias, batch_size)[0]
    running_mean = builder.call("Tile", running_mean, batch_size)[0]
    running_var = builder.call("Tile", running_var, batch_size)[0]

    norm = builder.call(
        "BatchNormalization",
        bn_input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=1.0 - momentum,
        training_mode=False,
        _num_outputs=1,
    )[0]
    return builder.call(
        "Reshape", norm, builder.call("Shape", input)[0], allowzero=True
    )[0]


@onnx_impl(aten.layer_norm.default)
def aten_layer_norm(
    input: ir.Value,
    normalized_shape: Sequence[int | ir.Value],
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
    cudnn_enable: bool = True,
) -> ir.Value:
    """layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"""
    builder = current_builder()

    # Use Python metadata to obtain start_axis.
    start_axis = -len(normalized_shape)

    if weight is None:
        one = builder.call("Constant", value=ir.tensor(1, dtype=input.dtype))[0]
        weight = builder.call(
            "Expand", one, builder.call("Shape", input, start=start_axis)[0]
        )[0]

    result, _, _ = builder.call(
        "LayerNormalization",
        input,
        weight,
        bias,
        axis=start_axis,
        epsilon=eps,
        _num_outputs=3,
    )
    return result


@onnx_impl(aten._native_batch_norm_legit_no_training.default)
def aten__native_batch_norm_no_training(
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
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )


@onnx_impl(aten._native_batch_norm_legit.no_stats)
def aten__native_batch_norm_no_stats(
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""

    return aten_native_batch_norm(
        input, weight, bias, None, None, training, momentum, eps
    )


@onnx_impl((aten.native_batch_norm.default, aten._native_batch_norm_legit.default))
def aten_native_batch_norm(
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
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )[:3]


def _aten_native_batch_norm_training_onnx(
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
    builder = current_builder()
    norm, running_mean, _ = builder.call(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=momentum,
        training_mode=True,
        _num_outputs=3,
    )
    # Compute mean and rstd
    # Mean, var, and rstd computation and results are expected to be
    # in higher precision when inputs are float16.
    upcast_input = builder.call(
        "Cast",
        input,
        to=ir.DataType.DOUBLE
        if input.dtype == ir.DataType.DOUBLE
        else ir.DataType.FLOAT,
    )[0]
    mean = builder.call("ReduceMean", upcast_input, axes)[0]
    input_sub_mean = builder.call("Sub", upcast_input, mean)[0]
    sqr = builder.call("Mul", input_sub_mean, input_sub_mean)[0]
    var = builder.call("ReduceMean", sqr, axes, keepdims=False)[0]
    rstd = builder.call(
        "Div", 1.0, builder.call("Sqrt", builder.call("Add", var, eps)[0])[0]
    )[0]
    # Get mean again with size = [1, C]
    mean = builder.call("ReduceMean", upcast_input, axes, keepdims=False)[0]

    # Compute the running var the PyTorch way
    # https://github.com/pytorch/pytorch/blob/5cc511f72fe073bbd8c10d796d72dce67f5cd5c4/torch/_decomp/decompositions.py#L1646

    n = builder.call(
        "Cast",
        builder.call(
            "Div",
            builder.call("Size", input)[0],
            builder.call("Gather", builder.call("Shape", input)[0], 1, axis=0)[0],
        )[0],
        to=upcast_input.dtype,
    )[0]
    unbiased_var = builder.call(
        "Mul", var, builder.call("Div", n, builder.call("Sub", n, 1.0)[0])[0]
    )[0]

    # NOTE: momentum in ONNX is 1.0-momentum in PyTorch
    new_running_var = builder.call(
        "Add",
        builder.call(
            "CastLike",
            builder.call("Mul", 1.0 - momentum, unbiased_var)[0],
            running_var,
        )[0],
        builder.call("Mul", momentum, running_var)[0],
    )[0]

    return norm, mean, rstd, running_mean, new_running_var


def _aten_native_batch_norm_inference_onnx(
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
    builder = current_builder()
    norm = builder.call(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=momentum,
        training_mode=False,
        _num_outputs=1,
    )[0]
    # CUDA and CPU gives different shapes:
    # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1451-L1457
    # We use CUDA's output here
    invstd = builder.call(
        "Div", 1.0, builder.call("Sqrt", builder.call("Add", running_var, eps)[0])[0]
    )[0]
    # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1475
    running_mean_fp32 = builder.call("Cast", running_mean, to=ir.DataType.FLOAT)[0]
    invstd = builder.call("Cast", invstd, to=ir.DataType.FLOAT)[0]
    return (
        norm,
        running_mean_fp32,
        invstd,
        builder.call("Identity", running_mean)[0],
        builder.call("Identity", running_var)[0],
    )


@onnx_impl(aten._native_batch_norm_legit_functional.default)
def aten__native_batch_norm_legit_functional(
    input: ir.Value,
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    running_mean: ir.Value | None = None,
    running_var: ir.Value | None = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
    builder = current_builder()
    if weight is None:  # Set to 1.0 as default
        weight = builder.call(
            "Expand",
            builder.constant([1.0], dtype=input.dtype),
            builder.call("Shape", input, start=1, end=2)[0],
        )[0]

    if bias is None:  # Set to 0.0 as default
        bias = builder.call(
            "Expand",
            builder.constant([0.0], dtype=input.dtype),
            builder.call("Shape", input, start=1, end=2)[0],
        )[0]

    axes = list(range(len(input.shape)))
    axes.pop(1)
    axes = builder.constant(axes, dtype=ir.DataType.INT64)
    if running_mean is None:  # Using input mean
        running_mean = builder.call("ReduceMean", input, axes, keepdims=False)[0]

    if running_var is None:  # Using input var
        mean = builder.call("ReduceMean", input, axes)[0]
        input_sub_mean = builder.call("Sub", input, mean)[0]
        sqr_input_sub_mean = builder.call("Mul", input_sub_mean, input_sub_mean)[0]
        running_var = builder.call(
            "ReduceMean", sqr_input_sub_mean, axes, keepdims=False
        )[0]

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
    builder = current_builder()

    # Actually we don't need N,C,HxW value because the input tensor has that information
    if weight is None:  # Set to 1.0 as default, the shape is Channel size
        weight = builder.call(
            "Expand",
            builder.constant([1.0], dtype=ir.DataType.FLOAT),
            builder.call("Shape", input, start=1, end=2)[0],
        )[0]

    if bias is None:  # Set to 0.0 as default, the shape is Channel size
        bias = builder.call(
            "Expand",
            builder.constant([0.0], dtype=ir.DataType.FLOAT),
            builder.call("Shape", input, start=1, end=2)[0],
        )[0]

    # Because onnx.GroupNorm() need size=group for weight and bias
    # But the torch's aten function's input need size=channel, the size mismatched
    # So we have to use onnx.InstanceNorm() to simulate
    # This implementation should be simplified after opset 21
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    # Create weight_instance_norm and bias_instance_norm, copied from Torch ONNX converter
    group_tensor = builder.call("Reshape", group, neg_1)[0]
    # 0 in the shape list keeps dimension value unchanged, for InstanceNorm need [0,group,-1]
    shape_input = builder.call(
        "Concat",
        builder.constant([0], dtype=ir.DataType.INT64),
        group_tensor,
        neg_1,
        axis=0,
    )[0]
    input_reshaped = builder.call("Reshape", input, shape_input)[0]
    weight_inst_norm = builder.call(
        "Expand", builder.call("CastLike", 1.0, input)[0], group_tensor
    )[0]
    bias_inst_norm = builder.call(
        "Expand", builder.call("CastLike", 0.0, input)[0], group_tensor
    )[0]
    norm = builder.call(
        "InstanceNormalization",
        input_reshaped,
        weight_inst_norm,
        bias_inst_norm,
        epsilon=eps,
    )[0]
    # Reshape back to input's shape
    norm = builder.call(
        "Reshape", norm, builder.call("Shape", input)[0], allowzero=True
    )[0]
    # Using the input weight and bias to do affine
    # But need to unsqueeze to the target shape for broading cast easy
    input_rank = len(input.shape)
    axes_unsqueeze = builder.call("Range", 1, input_rank - 1, 1)[0]
    weight_full_shape = builder.call("Unsqueeze", weight, axes_unsqueeze)[0]
    bias_full_shape = builder.call("Unsqueeze", bias, axes_unsqueeze)[0]
    weight_full_shape = builder.call("CastLike", weight_full_shape, norm)[0]
    norm_mul_weight = builder.call("Mul", norm, weight_full_shape)[0]
    bias_full_shape = builder.call("CastLike", bias_full_shape, norm_mul_weight)[0]
    norm_result = builder.call("Add", norm_mul_weight, bias_full_shape)[0]
    # Compute mean and rstd, but using Torch algorithm
    # The returned shape for mean and vstd should be [N, group, -1]
    N = builder.call("Shape", input, start=0, end=1)[0]
    shape_N_group_neg1 = builder.call("Concat", N, group_tensor, neg_1, axis=0)[0]
    input_N_group_neg1 = builder.call("Reshape", input, shape_N_group_neg1)[0]
    # The output size is [N, group], so dims = [2]
    axes = builder.constant([2], dtype=ir.DataType.INT64)
    # Get mean which size is [N, group, 1], for broadcasting
    mean = builder.call("ReduceMean", input_N_group_neg1, axes)[0]
    input_sub_mean = builder.call("Sub", input_N_group_neg1, mean)[0]
    sqr_input_sub_mean = builder.call("Mul", input_sub_mean, input_sub_mean)[0]
    # In Pytorch, vstd = 1/(sqrt(var + eps))
    var = builder.call("ReduceMean", sqr_input_sub_mean, axes, keepdims=False)[0]
    eps = builder.call("Constant", value=ir.tensor(eps, dtype=input.dtype))[0]
    one = builder.call("Constant", value=ir.tensor(1.0, dtype=input.dtype))[0]
    rstd = builder.call(
        "Div", one, builder.call("Sqrt", builder.call("Add", var, eps)[0])[0]
    )[0]
    # Get the correct shape [N, group] for mean again
    mean = builder.call("ReduceMean", input_N_group_neg1, axes, keepdims=False)[0]
    return norm_result, mean, rstd


@onnx_impl(aten.native_layer_norm.default)
def aten_native_layer_norm(
    input: ir.Value,
    normalized_shape: Sequence[int],
    weight: ir.Value | None = None,
    bias: ir.Value | None = None,
    eps: float = 1e-05,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"""
    builder = current_builder()

    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    # The mean and standard-deviation are calculated over the last D dimensions,
    # where D is the dimension of normalized_shape. For example, if normalized_shape is
    # (3, 5) (a 2-dimensional shape), the mean and standard-deviation are computed
    # over the last 2 dimensions of the input (i.e. input.mean((-2, -1))).

    # Use Python to manipulate
    start_axis = -len(normalized_shape)

    if weight is None:
        one = builder.constant([1.0], dtype=ir.DataType.FLOAT)
        weight = builder.call(
            "Expand", one, builder.call("Shape", input, start=start_axis)[0]
        )[0]
        weight = builder.call("CastLike", weight, input)[0]

    result, mean, rdenominator = builder.call(
        "LayerNormalization",
        input,
        weight,
        bias,
        axis=start_axis,
        epsilon=eps,
        _num_outputs=3,
    )

    return result, mean, rdenominator
