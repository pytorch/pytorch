#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace xnnpack {

//
// Pooling
//

bool use_max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode);

Tensor max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode);

} // namespace xnnpack
} // namespace native
} // namespace at
