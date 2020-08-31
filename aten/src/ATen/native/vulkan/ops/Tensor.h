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
  vTensor(IntArrayRef sizes, const TensorOptions& options);

  api::Resource::Buffer buffer();
  api::Resource::Image image();

 private:
  c10::SmallVector<int64_t, 4u> sizes_;
  TensorOptions options_;
  api::Resource::Buffer staging_;
  api::Resource::Buffer buffer_;
  api::Resource::Image image_;
};

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;
void verify(const TensorOptions& options);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
