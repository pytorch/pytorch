#pragma once

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor& copy_(Tensor& self, const Tensor& src);

#ifndef USE_VULKAN

inline Tensor& copy_(Tensor& self, const Tensor& src) {
  AT_ERROR("Vulkan backend was not linked to the build!");
}

#endif /* USE_VULKAN */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
