# copied from https://github.com/pytorch/ao/blob/main/torchao/quantization/observer.py
# and https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py
# PLEASE DON'T MODIFY THIS FILE SO THAT WE DON'T GET OUT OF SYNC
import logging
from abc import ABCMeta
from typing import Any, Optional, Union

import torch
from torch.ao.quantization.observer import (
    AffineQuantizedObserverBase,
    get_block_size,
    Granularity,
    MappingType,
    TorchAODType,
    ZeroPointDomain,
)


ABC: Any = ABCMeta("ABC", (object,), {})  # compatible with Python 2 *and* 3:

logger = logging.getLogger(__name__)

FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}
_SUB_BYTE_UINT_BOUNDS = {
    torch.uint1: (0, 2**1 - 1),
    torch.uint2: (0, 2**2 - 1),
    torch.uint3: (0, 2**3 - 1),
    torch.uint4: (0, 2**4 - 1),
    torch.uint5: (0, 2**5 - 1),
    torch.uint6: (0, 2**6 - 1),
    torch.uint7: (0, 2**7 - 1),
}

"""
Map from dtype to the bound value of integers
TODO: maybe can replace this with call to torch.iinfo
"""
_DTYPE_TO_QVALUE_BOUNDS: dict[Union[torch.dtype, TorchAODType], tuple[int, int]] = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
}
_DTYPE_TO_QVALUE_BOUNDS.update(_SUB_BYTE_UINT_BOUNDS)


def _is_float8_type(dtype: torch.dtype) -> bool:
    fp8_types = {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
    return dtype in fp8_types


# TODO: decide on if we want to allow custom quant_min/quant_max here
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
    if dtype in FP8_TYPES:
        quant_min_lower_bound, quant_max_upper_bound = (
            torch.finfo(dtype).min,
            torch.finfo(dtype).max,
        )
    elif dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    else:
        quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, (
        "quant_min out of bound for dtype, "
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
    )

    assert quant_max <= quant_max_upper_bound, (
        "quant_max out of bound for dtype, "
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    )
    return quant_min, quant_max


def _get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, (
                f"Expecting input size at {i} dimension: "
                f"{input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            )
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims


def _register_custom_op(lib):
    """This decorator is used to preserve some high level operators for torch.export.export
    while still allow them to be decomposed for inductor path

    requirement: make sure `fn.__name__[1:]` is the operator name you want to register

    NOTE: This should be applied at the top, after all other decorators have been applied
    NOTE: We haven't tested the case when `fn` accepts tensor subclass instance as input,
    e.g. uint4 tensor subclass instance, and we'll probably need to figure out what would make
    sense for downstream system (like executorch) to accept as well

    Example:
        lib = torch.library.Library("my_namespace', "FRAGMENT")

        register_custom_op = _register_custom_op(lib)

        @register_custom_op
        def _the_op_that_needs_to_be_preserved(...)
            ...

        # after this, `_the_op_that_needs_to_be_preserved` will be preserved as
        # torch.ops.my_namespace.the_op_that_needs_to_be_preserved operator after
        # torch.export.export / torch._export.export_for_training

    """
    from torch._inductor.decomposition import register_decomposition

    def decorator(fn):
        from torch._library.infer_schema import infer_schema

        # expecting fn.__name__ starts with `_` and we want to take the rest
        # to be the name of the custom op
        assert fn.__name__[0] == "_", (
            f"Expecting function name starts with `_`, got {fn.__name__}"
        )
        assert not any(c in fn.__name__ for c in ".<>"), (
            f"Expecting op to be defined in normal functions, not lambda or local: {fn.__name__}"
        )
        op_name = fn.__name__[1:]
        schema = op_name + infer_schema(fn, mutates_args={})
        lib.define(schema)
        lib.impl(op_name, fn, "CompositeImplicitAutograd")

        lib_namespace = lib.ns
        op = getattr(getattr(torch.ops, lib_namespace), op_name)
        register_decomposition([op])(fn)
        return op

    return decorator


quant_lib = torch.library.Library("pt2e_quant", "FRAGMENT")  # noqa: TOR901

register_custom_op = _register_custom_op(quant_lib)


def choose_qparams_affine_with_min_max(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    mapping_type: MappingType,
    block_size: tuple[int, ...],
    target_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A variant of :func:`~torchao.quantization.quant_primitives.choose_qparams_affine`
    operator that pass in min_val and max_val directly instead of deriving these from a single input.
    This is used for observers in static quantization where min_val and max_val may be obtained through
    tracking all the data in calibration data set.

    Args:
      Mostly same as :func:`~torchao.quantization.quant_primitives.choose_qparams_affine`. with one
      difference: instead of passing in `input` Tensor and use that to calculate min_val/max_val
      and then scale/zero_point, we pass in min_val/max_val directly
    """
    return _choose_qparams_affine(
        None,
        mapping_type.name,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype,
        zero_point_dtype,
        preserve_zero,
        zero_point_domain.name if zero_point_domain is not None else None,
        min_val,
        max_val,
    )


@register_custom_op
def _choose_qparams_affine(
    input: Optional[torch.Tensor],
    mapping_type: str,
    block_size: list[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    preserve_zero: bool = True,
    zero_point_domain: Optional[str] = "INT",
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """op definition that has compatible signatures with custom op library

    The op does the following:
    1. figure out the dimension for reduction based on block_size
    2. find min_val/max_val based on the dimension for reduction
    3. calculate quantization parameters based on min_val/max_val based on args like `preserve_zero`
       and `zero_point_domain`
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [
        MappingType.SYMMETRIC.name,
        MappingType.SYMMETRIC_NO_CLIPPING_ERR.name,
        MappingType.ASYMMETRIC.name,
    ], f"Unsupported mapping type: {mapping_type}"
    if target_dtype in FP8_TYPES:
        assert mapping_type == MappingType.SYMMETRIC.name, (
            f"Only symmetric quantization is supported for FP8 types, got {mapping_type}"
        )

    if input is not None:
        if scale_dtype is None:
            scale_dtype = input.dtype
        if zero_point_dtype is None:
            zero_point_dtype = input.dtype
        if eps is None:
            eps = torch.finfo(input.dtype).eps

        assert len(block_size) == input.dim(), (
            f"Got input dim:{input.dim()}, block_size: {block_size}"
        )
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, input.size()
        )
        input = input.view(shape_for_reduction)

        min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input, dim=reduction_dims, keepdim=False)
    else:
        assert min_val is not None and max_val is not None, (
            f"Need to provide `min_val` and `max_val` when `input` is None, got: {min_val, max_val}"
        )
        assert min_val.dtype == max_val.dtype, (
            f"Expecting `min_val` and `max_val` to have the same dtype, got: {min_val.dtype, max_val.dtype}"
        )

        if scale_dtype is None:
            scale_dtype = min_val.dtype
        if zero_point_dtype is None:
            zero_point_dtype = min_val.dtype
        if eps is None:
            eps = torch.finfo(min_val.dtype).eps

    if preserve_zero:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    else:
        min_val_neg = min_val
        max_val_pos = max_val

    if (
        mapping_type == MappingType.SYMMETRIC.name
        or mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR.name
    ):
        # scales
        if mapping_type == MappingType.SYMMETRIC.name:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
        else:
            assert mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR.name
            # calculate smin and smax individually and choose the larger one. For example, if quant_min = -8 and
            # quant_max = 7.
            # - If smin is bigger: There would be coverage on negative values down to -8, and less rounding
            # error than the existing SYMMETRIC case.
            # - If smax is bigger: it covers the positive values up to 7. The round
            # error may be bigger than the existing SYMMETRIC case. Either way, there's no out-of-range fp values after
            # quantization.
            smin = min_val_neg / float(quant_min)
            smax = max_val_pos / float(quant_max)
            mask = smin > smax
            scale = torch.where(mask, smin, smax)
        # zeros
        if not preserve_zero:
            raise ValueError(
                "preserve_zero == False is not supported for symmetric quantization"
            )
        if (
            zero_point_domain is not None
            and zero_point_domain != ZeroPointDomain.INT.name
        ):
            raise ValueError(
                "zero_point_domain != ZeroPointDomain.INT is not supported for symmetric quantization"
            )
        scale = torch.clamp(scale, min=eps)
        zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
    else:
        assert mapping_type == MappingType.ASYMMETRIC.name
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.clamp(scale, min=eps)
        if zero_point_domain == ZeroPointDomain.NONE.name:
            zero_point = None
        else:
            if preserve_zero:
                zero_point = quant_min - torch.round(min_val_neg / scale)
                zero_point = torch.clamp(zero_point, quant_min, quant_max)
            else:
                assert zero_point_domain == ZeroPointDomain.FLOAT.name, (
                    "if not preserve_zero, zero_point must be in FLOAT domain"
                )
                mid_point = (quant_max + quant_min + 1) / 2
                zero_point = min_val_neg + scale * mid_point

    if zero_point is not None:
        zero_point = zero_point.to(dtype=zero_point_dtype)
    return scale.to(dtype=scale_dtype), zero_point


@torch.no_grad()
def quantize_affine(
    input: torch.Tensor,
    block_size: tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): original float32, float16 or bfloat16 Tensor
      block_size: (Tuple[int, ...]): granularity of quantization,
           this means the size of the tensor elements that's sharing the same qparam
           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      output_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
      quantized tensor with requested dtype
    """
    return _quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        output_dtype,
        quant_min,
        quant_max,
        zero_point_domain.name if zero_point_domain is not None else None,
    )


@register_custom_op
def _quantize_affine(
    input: torch.Tensor,
    block_size: list[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    zero_point_domain: Optional[str] = ZeroPointDomain.INT.name,
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library

    Note:
        zero_point_domain is optional specifies how we quantize the floating point to quantized data:
        INT: quantized_val = (float_val / scale) (integer) + zero_point (integer)
        FLOAT: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
        None: quantized_val = (float_val / scale) | this is primarily used for floatx quantization
            Where we do not want to round values to nearest integer and instead scale and cast.
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    # workaround for uintx dtypes, since we don't have native Uintx dtype connected with
    # torch.uintx dtypes yet
    if output_dtype in _SUB_BYTE_UINT_BOUNDS:
        output_dtype = torch.uint8
    return _quantize_affine_no_dtype_cast(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        zero_point_domain,
    ).to(output_dtype)


def _quantize_affine_no_dtype_cast(
    input: torch.Tensor,
    block_size: list[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
    zero_point_domain: Optional[str] = ZeroPointDomain.INT.name,
) -> torch.Tensor:
    """
    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. quantize the input based on the quantization parameters scale and zero_point and args like zero_point_domain
    3. reshape the quantized result to original shape
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT.name:
        quant = torch.clamp(
            torch.round(input * (1.0 / scale)) + zero_point, quant_min, quant_max
        )
    elif zero_point_domain == ZeroPointDomain.NONE.name:
        assert zero_point is None, (
            "zero_point should be None when zero_point_domain is NONE"
        )
        quant = torch.clamp(torch.round(input * (1.0 / scale)), quant_min, quant_max)
    elif zero_point_domain is None:
        # This case handles quantization for float8 we expect no zero point and no zero point domain
        assert zero_point is None, (
            "zero_point should be None when zero_point_domain is None"
        )
        quant = torch.clamp(input * scale.reciprocal(), quant_min, quant_max)
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT.name
        mid_point = (quant_max + quant_min + 1) / 2
        min_val = zero_point - scale * mid_point
        quant = torch.clamp(
            torch.round((input - min_val) / scale), quant_min, quant_max
        )
    quant = quant.view(original_shape)

    return quant


def dequantize_affine(
    input: torch.Tensor,
    block_size: tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization,
        this means the size of the tensor elements that's sharing the same qparam
        e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization
      input_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be either integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
    return _dequantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        input_dtype,
        quant_min,
        quant_max,
        zero_point_domain.name if zero_point_domain is not None else None,
        output_dtype=output_dtype,
    )


@register_custom_op
def _dequantize_affine(
    input: torch.Tensor,
    block_size: list[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[Union[int, float, bool]] = None,
    quant_max: Optional[Union[int, float, bool]] = None,
    zero_point_domain: Optional[str] = ZeroPointDomain.INT.name,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """op definition that has compatible signatures with custom op library"""
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    if input_dtype not in _SUB_BYTE_UINT_BOUNDS:
        assert input.dtype == input_dtype, (
            f"Expected: {input_dtype}, got: {input.dtype}"
        )
    assert output_dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported output dtype: {output_dtype}"
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)
    return _dequantize_affine_no_dtype_check(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        zero_point_domain,
        output_dtype,
    )


def _dequantize_affine_no_dtype_check(
    input: torch.Tensor,
    block_size: list[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
    zero_point_domain: Optional[str] = ZeroPointDomain.INT.name,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function converts AQT tensors to their high precision floating point representation

    The op does the following:
    1. figure out the dimension for reduction based on block_size, also reshape the input to align with
       the shape after reduction
    2. dequantize the input based on the quantization parameters scale and zero_point and args like zero_point_domain
    3. reshape the quantized result to original shape and change dtype to the output_dtype
    """
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT.name:
        # Force a copy to avoid input modification due
        # to upcoming in-place operations.
        dequant = input.to(torch.int32, copy=True)
        if zero_point is not None:
            dequant = dequant - zero_point.to(torch.int32)
        dequant = dequant.to(output_dtype)
        dequant = dequant * scale
    elif zero_point_domain == ZeroPointDomain.NONE.name:
        assert zero_point is None, (
            "zero_point should be None when zero_point_domain is NONE"
        )
        dequant = input.to(output_dtype)
        dequant = dequant * scale
    elif zero_point_domain is None:
        # This case handles dequantization for float8 we expect no zero point and no zero point domain
        assert zero_point is None, (
            "zero_point should be None when zero_point_domain is None"
        )
        assert _is_float8_type(input.dtype), (
            f"dequantiztion with no zero point domain is only supported with FP8 types, got {input.dtype}"
        )
        dequant = input.to(output_dtype)
        dequant = dequant * scale
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT.name, (
            f"Unexpected zero point domain: {zero_point_domain}"
        )
        # TODO: this seems to be a detail for tinygemm (converting from uint to int, probably need to refactor this)
        mid_point = (quant_max + quant_min + 1) / 2
        # This should allocate new memory and avoid input modification
        dequant = input - mid_point
        dequant = dequant.to(output_dtype)
        dequant *= scale
        if zero_point is not None:
            dequant += zero_point

    return dequant.view(original_shape).to(output_dtype)


class AffineQuantizedMinMaxObserver(AffineQuantizedObserverBase):
    def forward(self, input: torch.Tensor):
        if input.numel() == 0:
            return input

        input_detached = input.detach()
        self.original_dtype = input_detached.dtype
        assert self.granularity is not None, "granularity is None"
        self.block_size = get_block_size(input_detached.shape, self.granularity)

        shape_for_reduction, reduction_dims = _get_reduction_params(
            self.block_size, input_detached.size()
        )
        input_detached = input_detached.view(shape_for_reduction)
        min_val = torch.amin(input_detached, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input_detached, dim=reduction_dims, keepdim=False)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert self.min_val.shape == min_val.shape, (
                f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            )
            assert self.max_val.shape == max_val.shape, (
                f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            )
            min_val = torch.min(self.min_val, min_val)
            max_val = torch.max(self.max_val, max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        # returning original input
        return input

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert hasattr(self, "min_val") and hasattr(self, "max_val"), (
            "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        )
        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [],  # BlockSize is not needed because the min/max are already reduced
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )


class AffineQuantizedMovingAverageMinMaxObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        averaging_constant=0.01,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        is_dynamic=False,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        # there could be some extra args that's ignored
        **kwargs,
    ):
        self.is_dynamic = is_dynamic
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(
                "MovingAverageMinMaxObserver doesn't support dynamic quantization for "
                f"averaging constant of {self.averaging_constant}"
            )

        super().__init__(
            mapping_type=mapping_type,
            target_dtype=target_dtype,
            granularity=granularity,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
        )

    def forward(self, input: torch.Tensor):
        if input.numel() == 0:
            return input

        input_detached = input.detach()
        self.original_dtype = input_detached.dtype
        assert self.granularity is not None, "granularity is None"
        self.block_size = get_block_size(input_detached.shape, self.granularity)

        shape_for_reduction, reduction_dims = _get_reduction_params(
            self.block_size, input_detached.size()
        )
        input_detached = input_detached.view(shape_for_reduction)
        min_val = torch.amin(input_detached, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input_detached, dim=reduction_dims, keepdim=False)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert self.min_val.shape == min_val.shape, (
                f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            )
            assert self.max_val.shape == max_val.shape, (
                f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            )
            min_val = self.min_val + self.averaging_constant * (min_val - self.min_val)
            max_val = self.max_val + self.averaging_constant * (max_val - self.max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)

        # returning original input
        return input

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert hasattr(self, "min_val") and hasattr(self, "max_val"), (
            "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        )

        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [],  # BlockSize is not needed because the min/max are already reduced
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )


class AffineQuantizedPlaceholderObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        is_dynamic=False,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        # there could be some extra args that's ignored
        **kwargs,
    ):
        self.is_dynamic = is_dynamic

        super().__init__(
            mapping_type=mapping_type,
            target_dtype=target_dtype,
            granularity=granularity,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
        )

    def forward(self, input):
        self.block_size = get_block_size(input.shape, self.granularity)
        self.original_dtype = input.dtype
        return input

    def calculate_qparams(self):
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for PlaceholderObserver"
        )
