#pragma once

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor& copy_(Tensor& self, const Tensor& src);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
