#pragma once

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <cmath>

/* FakeQuantize Op for PerChannelAffine quantization scheme */
namespace at {
namespace native {
void fake_quantize_slice_cuda(
    Tensor& output,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

void fake_quantize_grad_slice_cuda(
    Tensor& input_grad,
    const Tensor& output_grad,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max);

} // namespace native
} // namespace at
