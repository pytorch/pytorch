#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

struct Persistent final {
  typedef api::Handle<
      api::Resource::Buffer,
      std::function<void(const api::Resource::Buffer&)>> Buffer;

  typedef api::Handle<
      api::Resource::Image,
      std::function<void(const api::Resource::Image&)>> Image;

  class Pool final {
   public:
    explicit Pool(const api::GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = default;
    Pool& operator=(Pool&&) = default;
    ~Pool() = default;

    Buffer buffer(
        const api::Resource::Buffer::Descriptor& descriptor,
        c10::ArrayRef<const uint8_t> data);

    Image image(
        const api::Resource::Image::Descriptor& descriptor,
        c10::ArrayRef<const uint8_t> data);

   private:
    api::Resource::Pool pool_;
  };
};

Persistent::Pool* persistent();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
