#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace xnnpack {

//
// Convolution
//

bool use_convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups);

Tensor convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups);

//
// Max Pooling
//

bool use_max_pool2d(
    const Tensor& input,
    IntArrayRef kernel,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    bool ceil_mode,
    float output_min = -std::numeric_limits<float>::infinity(),
    float output_max = +std::numeric_limits<float>::infinity());

Tensor max_pool2d(
    const Tensor& input,
    IntArrayRef kernel,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    bool ceil_mode,
    float output_min = -std::numeric_limits<float>::infinity(),
    float output_max = +std::numeric_limits<float>::infinity());

} // namespace xnnpack
} // namespace native
} // namespace at
