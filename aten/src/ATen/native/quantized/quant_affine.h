#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

Tensor quantize_tensor_affine(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
Tensor quantize_tensor_per_channel_affine(Tensor qtensor,
                                          Tensor rtensor,
                                          Tensor scales,
                                          Tensor zero_points,
                                          int64_t axis);

Tensor dequantize_tensor_affine(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point);
Tensor dequantize_tensor_per_channel_affine(Tensor qtensor,
                                            Tensor rtensor,
                                            Tensor scales,
                                            Tensor zero_points,
                                            int64_t axis);

using quantize_tensor_affine_fn = void (*)(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point);

using quantize_tensor_per_channel_affine_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

using dequantize_tensor_affine_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    double scale,
    int64_t zero_point);

using dequantize_tensor_per_channel_affine_fn = void (*)(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

DECLARE_DISPATCH(quantize_tensor_affine_fn, quantize_tensor_affine_stub);
DECLARE_DISPATCH(quantize_tensor_per_channel_affine_fn, quantize_tensor_per_channel_affine_stub);

DECLARE_DISPATCH(dequantize_tensor_affine_fn, dequantize_tensor_affine_stub);
DECLARE_DISPATCH(dequantize_tensor_per_channel_affine_fn, dequantize_tensor_per_channel_affine_stub);

} // namespace native
} // namespace at
