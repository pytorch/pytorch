#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor add(
    const Tensor& self,
    const Tensor& other,
    const Scalar alpha) {

  auto xt = self.is_vulkan() ? self : self.vulkan();

  return Tensor{};
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("add.Tensor", TORCH_FN(at::native::vulkan::ops::add));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
