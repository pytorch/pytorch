#include <ATen/native/vulkan/ops/Common.h>

#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Scalar _local_scalar_dense(const Tensor& self) {
  TORCH_CHECK(
      self.dtype() == ScalarType::Float, "Only float dtype is supported");
  return Scalar(self.cpu().item<float>());
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_local_scalar_dense"),
      TORCH_FN(_local_scalar_dense));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
