#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class Pool final {
 public:
  explicit Pool(const api::GPU& gpu);
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  Pool(Pool&&) = default;
  Pool& operator=(Pool&&) = default;
  ~Pool() = default;

  api::Resource::Buffer buffer(c10::ArrayRef<const uint8_t> data);
  api::Resource::Image image(c10::ArrayRef<const uint8_t> data);

 private:
  api::Resource::Pool pool_;
};

Pool* persistent();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
