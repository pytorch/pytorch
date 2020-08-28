#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Resource.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class vTensor final {
 public:
  vTensor();
  explicit vTensor(IntArrayRef sizes);
  // vTensor(const vTensor&) = delete;
  // vTensor& operator=(const vTensor&) = delete;
  ~vTensor() = default;

 private:
  Resource::Image image_;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
