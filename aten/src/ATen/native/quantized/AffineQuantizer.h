#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

namespace at::native {

TORCH_API Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point);
TORCH_API Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis);

TORCH_API Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_API Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point);
TORCH_API Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    Tensor zero_points,
    int64_t axis);
TORCH_API Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor& rtensor, Tensor& qtensor, double scale, int64_t zero_point);

using quantize_tensor_per_channel_affine_fn = void (*)(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_channel_float_qparams_fn = void (*)(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using dequantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor& qtensor, Tensor& rtensor, double scale, int64_t zero_point);

using dequantize_tensor_per_channel_affine_fn = void (*)(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using dequantize_tensor_per_channel_float_qparams_fn = void (*)(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor& rtensor, Tensor& qtensor, float scale, float zero_point);

using dequantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor& qtensor, Tensor& rtensor, float scale, float zero_point);

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_fn,
    quantize_tensor_per_tensor_affine_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_channel_affine_fn,
    quantize_tensor_per_channel_affine_stub)
DECLARE_DISPATCH(
    quantize_tensor_per_channel_float_qparams_fn,
    quantize_tensor_per_channel_float_qparams_stub)

DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_fn,
    dequantize_tensor_per_tensor_affine_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_affine_fn,
    dequantize_tensor_per_channel_affine_stub)
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_fn,
    dequantize_tensor_per_channel_float_qparams_stub)

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_fn,
    quantize_tensor_per_tensor_affine_sub_byte_stub)

DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_fn,
    dequantize_tensor_per_tensor_affine_sub_byte_stub)

template <typename T>
TORCH_API Tensor quantize_tensor(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point);
template <typename T>
TORCH_API Tensor dequantize_tensor(
    Tensor qtensor,
    Tensor rtensor,
    double scale,
    int64_t zero_point);

} // namespace at
