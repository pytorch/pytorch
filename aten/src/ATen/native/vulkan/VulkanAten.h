#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace vulkan {

Tensor convolution_prepack_weights(const at::Tensor& weight);

Tensor convolution_prepacked(
    const at::Tensor& input, // Vulkan
    IntArrayRef weightSizes,
    const at::Tensor& weight_prepacked_vulkan, // Vulkan
    const c10::optional<at::Tensor>& bias, // Vulkan|CPU
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    const float output_min,
    const float output_max);

} // namespace vulkan
} // namespace native
} // namespace at
