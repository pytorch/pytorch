#pragma once

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class vTensor final {
 public:
  vTensor();
  explicit vTensor(IntArrayRef sizes);

 private:
  api::Resource::Image image_;
};

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
