#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

Tensor quantize_tensor_per_tensor_affine(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point);
Tensor quantize_tensor_per_channel_affine(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

Tensor quantize_tensor_per_channel_float_qparams(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

Tensor dequantize_tensor_per_tensor_affine(
    Tensor qtensor,
    Tensor rtensor,
    double scale,
    int64_t zero_point);
Tensor dequantize_tensor_per_channel_affine(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);
Tensor dequantize_tensor_per_channel_float_qparams(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_fn =
    void (*)(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);

using quantize_tensor_per_channel_affine_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

using quantize_tensor_per_channel_float_qparams_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

using dequantize_tensor_per_tensor_affine_fn =
    void (*)(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point);

using dequantize_tensor_per_channel_affine_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

using dequantize_tensor_per_channel_float_qparams_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

using quantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(Tensor rtensor, Tensor qtensor, float scale, float zero_point);

using dequantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(Tensor qtensor, Tensor rtensor, float scale, float zero_point);

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_fn,
    quantize_tensor_per_tensor_affine_stub);
DECLARE_DISPATCH(
    quantize_tensor_per_channel_affine_fn,
    quantize_tensor_per_channel_affine_stub);
DECLARE_DISPATCH(
    quantize_tensor_per_channel_float_qparams_fn,
    quantize_tensor_per_channel_float_qparams_stub);

DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_fn,
    dequantize_tensor_per_tensor_affine_stub);
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_affine_fn,
    dequantize_tensor_per_channel_affine_stub);
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_fn,
    dequantize_tensor_per_channel_float_qparams_stub);

DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_fn,
    quantize_tensor_per_tensor_affine_sub_byte_stub);

DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_fn,
    dequantize_tensor_per_tensor_affine_sub_byte_stub);

// Quantize a float value into a uint value given scale and zero_point
template <typename T>
CAFFE2_API T quantize_val(double scale, int64_t zero_point, float value);
// TODO combine this with quantize_val once the numerics for ARM are aligned
// with it
uint8_t quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value);
template <typename T, int precision = 8>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count = 8);
template <typename T>
CAFFE2_API Tensor quantize_tensor(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point);
template <typename T>
CAFFE2_API float dequantize_val(double scale, int64_t zero_point, T value);
template <typename T>
CAFFE2_API float dequantize_vec(
    double scale,
    int64_t zero_point,
    const T* src,
    float* dst,
    size_t count = 8);
template <typename T>
CAFFE2_API Tensor dequantize_tensor(
    Tensor qtensor,
    Tensor rtensor,
    double scale,
    int64_t zero_point);
template <typename SRC_T, typename DST_T>
CAFFE2_API DST_T requantize_val(double, int64_t, double, int64_t, SRC_T src);

// Given a multiplier and a zero_point, requantize int32_t computed values back
// to quantized values. See comment above
// make_per_tensor_affine_quantizer function for the usage of int64_t
template <typename DST_T>
CAFFE2_API DST_T
requantize_from_int(double multiplier, int64_t zero_point, int64_t src);

template <typename T>
CAFFE2_API T quantize_val_float_qparams(float scale, float zero_point, float value);

} // namespace native
} // namespace at
