# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math
from typing import Optional, Tuple

import torch
from torch._refs import _unsqueeze_multiple
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from torch.library import impl, Library


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

_INTEGER_DTYPES = [torch.uint8, torch.int8, torch.uint16, torch.int16, torch.int32]
_FLOAT_DTYPES = [torch.float8_e5m2, torch.float8_e4m3fn]

_DTYPE_TO_QVALUE_BOUNDS = {
    k: (torch.iinfo(k).min, torch.iinfo(k).max) for k in _INTEGER_DTYPES
}
_DTYPE_TO_QVALUE_BOUNDS.update(
    {k: (int(torch.finfo(k).min), int(torch.finfo(k).max)) for k in _FLOAT_DTYPES}
)


# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    assert quant_min >= quant_min_lower_bound, (
        "quant_min out of bound for dtype, "
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
    )

    assert quant_max <= quant_max_upper_bound, (
        "quant_max out of bound for dtype, "
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    )


quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    assert (
        input.dtype == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return torch.clamp(
        torch.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "Meta")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    assert (
        input.dtype == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert (
        zero_point.numel() == 1
    ), f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert (
        scale.numel() == 1
    ), f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return quantize_per_tensor(
        input, scale.item(), zero_point.item(), quant_min, quant_max, dtype
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    assert (
        zero_point.numel() == 1
    ), f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert (
        scale.numel() == 1
    ), f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    assert (
        input.dtype == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    return torch.empty_like(input, dtype=dtype)


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor2", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor2(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert (
        zero_point.numel() == 1
    ), f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert (
        scale.numel() == 1
    ), f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),
        quant_min.item(),
        quant_max.item(),
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "Meta")
def quantize_per_tensor_tensor2_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return quantize_per_tensor_tensor_meta(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    assert (
        input.dtype == dtype
    ), f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
    if out_dtype is None:
        out_dtype = torch.float32
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        return (input.to(out_dtype) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "Meta")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    return torch.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert (
        zero_point.numel() == 1
    ), f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert (
        scale.numel() == 1
    ), f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),
        quant_min,
        quant_max,
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    assert (
        zero_point.numel() == 1
    ), f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert (
        scale.numel() == 1
    ), f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=out_dtype)
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor2",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor2(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert (
        zero_point.numel() == 1
    ), f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert (
        scale.numel() == 1
    ), f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return dequantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),
        quant_min.item(),
        quant_max.item(),
        dtype,
        out_dtype=out_dtype,
    )


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "Meta")
def dequantize_per_tensor_tensor2_meta(
    input,
    scale,
    zero_point,
    quant_min,
    quant_max,
    dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return dequantize_per_tensor_tensor_meta(
        input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype
    )


quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
    assert (
        dtype in _DTYPE_TO_QVALUE_BOUNDS
    ), f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)

    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
    )


quantized_decomposed_lib.define(
    "choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_symmetric.tensor",
    "CompositeExplicitAutograd",
)
def choose_qparams_symmetric_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
    assert (
        dtype in _DTYPE_TO_QVALUE_BOUNDS
    ), f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=torch.per_tensor_symmetric,
    )


@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
def choose_qparams_tensor_meta(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
    assert (
        quant_min < quant_max
    ), f"Expecting quant_min to be smaller than quant_max but received min: \
        {quant_min} max: {quant_max}"
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(
        1, dtype=torch.int64, device=input.device
    )


@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "Meta")
def choose_qparams_symmetric_tensor_meta(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(
        1, dtype=torch.int64, device=input.device
    )


# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list


quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    assert (
        input.dtype == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    zero_points = zero_points.view(new_shape)

    res = torch.clamp(
        torch.round(input * (1.0 / scales)) + zero_points, quant_min, quant_max
    )
    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
def quantize_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    assert (
        input.dtype == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    assert (
        input.dtype == dtype
    ), f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    if out_dtype is None:
        out_dtype = torch.float32
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)

    new_shape = [1] * input.dim()
    new_shape[0] = scales.shape[0]
    scales = scales.view(new_shape)
    if zero_points is not None:
        res = (input - zero_points.view(new_shape)) * scales
    else:
        res = input * scales

    res = res.to(out_dtype)

    out = res.permute(tuple(permute_axis_list))
    return out


@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert (
        input.dtype == dtype
    ), f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    if out_dtype is None:
        out_dtype = torch.float32
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """

    scales = input.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    if dtype == torch.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        raise Exception(  # noqa: TRY002
            f"unsupported dtype in choose_qparams_per_token: {dtype}"
        )

    scales = scales.clamp(min=1e-5).div(quant_max)
    zero_points = torch.zeros_like(scales)
    return scales, zero_points


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = list(input.shape[:-1]) + [1]
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


quantized_decomposed_lib.define(
    "_choose_qparams_per_token_asymmetric_impl(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "_choose_qparams_per_token_asymmetric_impl",
    "CompositeImplicitAutograd",
)
def _choose_qparams_per_token_asymmetric_impl(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val = torch.amin(input, dim=-1, keepdim=True)
    max_val = torch.amax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(torch.float64), zero_point.to(torch.int64)


quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _choose_qparams_per_token_asymmetric_impl(input, dtype)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = list(input.shape[:-1]) + [1]
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    num_tokens = math.prod(list(input.size())[:-1])
    assert (
        num_tokens == scales.numel()
    ), f"num_tokens: {num_tokens} scales: {scales.size()}"
    assert (
        num_tokens == zero_points.numel()
    ), f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"


quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
def quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    input = (
        input.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp(quant_min, quant_max)
        .to(dtype)
    )
    return input


@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
def quantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
def dequantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float64 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int64 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
    input = input - zero_points
    input = input * scales
    # Since scales are of float64 type, we need to cast it to output dtype requested
    return input.to(output_dtype)


@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
def dequantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: support fp16
    return torch.empty_like(input, dtype=output_dtype)


quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# TODO: dtype is ignored for now
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
def quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    assert group_size > 1
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2

    # TODO: check for dtype, currently we can't express torch.int4 so it's omitted
    to_quant = input.reshape(-1, group_size)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    input_int8 = (
        to_quant.mul(1.0 / scales)
        .add(zero_points)
        .round()
        .clamp_(quant_min, quant_max)
        .to(dtype)
        .reshape_as(input)
    )

    return input_int8


@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
def quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    """Groupwise quantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    assert group_size > 1
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_group(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size: int = 128,
    output_dtype: torch.dtype = torch.float32,
):
    """Groupwise dequantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

    assert group_size > 1
    # needed for GPTQ single column dequantize
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    assert w_int8.shape[-1] % group_size == 0
    assert w_int8.dim() == 2

    w_int8_grouped = w_int8.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    if zero_points is not None:
        zp = zero_points.reshape(-1, 1)
    else:
        zp = torch.zeros([], dtype=torch.int32, device=scales.device)
    w_dq = w_int8_grouped.sub(zp).mul(scales).reshape_as(w_int8).to(output_dtype)
    return w_dq


quantized_decomposed_lib.define(
    "fake_quant_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max) -> Tensor"
)


class FakeQuantPerChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max):
        if scales.dtype != torch.float32:
            scales = scales.to(torch.float32)
        if zero_points.dtype != torch.int32:
            zero_points = zero_points.to(torch.int32)
        assert (
            input.dtype == torch.float32
        ), f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
        broadcast_dims = list(range(0, axis)) + list(range(axis + 1, input.ndim))
        unsqueeze_scales = _unsqueeze_multiple(scales, broadcast_dims)
        unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
        temp = torch.round(input * (1.0 / unsqueeze_scales)) + unsqueeze_zero_points
        out = (
            torch.clamp(temp, quant_min, quant_max) - unsqueeze_zero_points
        ) * unsqueeze_scales
        mask = torch.logical_and((temp >= quant_min), (temp <= quant_max))

        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Autograd")
def fake_quant_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    return FakeQuantPerChannel.apply(
        input, scales, zero_points, axis, quant_min, quant_max
    )


@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Meta")
def fake_quant_per_channel_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    return torch.empty_like(input)


quantized_decomposed_lib.define(
    "convert_element_type.no_fuse(Tensor input, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "convert_element_type.no_fuse",
    "CompositeExplicitAutograd",
)
def convert_element_type(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.ops.prims.convert_element_type.default(input, dtype)


@impl(quantized_decomposed_lib, "convert_element_type.no_fuse", "Meta")
def convert_element_type_meta(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)
