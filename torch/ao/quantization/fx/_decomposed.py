import torch
from torch.library import Library, impl
from torch.ao.quantization import MinMaxObserver

# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    quant_min_lower_bound = 0
    quant_max_upper_bound = 0
    if dtype == torch.uint8:
        quant_min_lower_bound = 0
        quant_max_upper_bound = 255
    elif dtype == torch.int8:
        quant_min_lower_bound = -128
        quant_max_upper_bound = 127
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"

quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(input, scale, zero_point, quant_min, quant_max, dtype):
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)

quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor("
    "Tensor input, Tensor scale, Tensor zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd")
def quantize_per_tensor_tensor(input, scale, zero_point, quant_min, quant_max, dtype):
    assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(input, scale, zero_point, quant_min, quant_max, dtype):
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    if dtype in [torch.uint8, torch.int8]:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        return (input.to(torch.float32) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor("
    "Tensor input, Tensor scale, Tensor zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor_tensor(input, scale, zero_point, quant_min, quant_max, dtype):
    assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
    return dequantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)


quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, ScalarType dtype) -> (Tensor, Tensor)")

@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(input, quant_min, quant_max, dtype):
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert quant_min < quant_max, f"Expecting quant_min to be smaller than quant_max but received min: {quant_min} max: {quant_max}"

    # Its weird to create an observer manually just to calculate qparams. I tried refactoring this functionality out of observer
    # into a util and then use that util directly, but I kept running into jit typing errors related to torch.qscheme not
    # being recognized as a type. TODO: properly refactor this out to avoid observer overhead
    tensor_dtype_to_observer_dtype = {torch.uint8: torch.quint8, torch.int8: torch.qint8}
    observer = MinMaxObserver(quant_min=quant_min, quant_max=quant_max, dtype=tensor_dtype_to_observer_dtype[dtype])
    observer(input)
    scale, zero_point = observer.calculate_qparams()
    return (scale, zero_point)

# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list

quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(input, scales, zero_points, axis, quant_min, quant_max, dtype):
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input)

    for i in range(input.size(0)):
        res[i] = torch.clamp(
            torch.round(input[i] * (1.0 / scales[i])) + zero_points[i],
            quant_min,
            quant_max
        )

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(input, scales, zero_points, axis, quant_min, quant_max, dtype):
    assert input.dtype == dtype, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input, dtype=torch.float32)

    for i in range(input.size(0)):
        # TODO: investigate why
        # (input[i] - zero_points[i]).to(torch.float32) * scales[i]
        # failed the test
        res[i] = (input[i].to(torch.float32) - zero_points[i]) * scales[i]

    out = res.permute(tuple(permute_axis_list))
    return out
