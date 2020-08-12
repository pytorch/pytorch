#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Command final {
  //
  // Buffer
  //

  class Buffer final {
   public:

   private:
    VkCommandBuffer command_buffer_;
  };

  //
  // Pool
  //

  class Pool final {
   public:

   private:
    VkCommandPool command_pool_;
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
