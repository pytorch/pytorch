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
  // Pool
  //

  struct Pool final {
    /*
      Descriptor
    */

    struct Descriptor final {
      uint32_t queue_family_index;

      inline bool operator==(const Descriptor& descriptor) const {
        return queue_family_index == descriptor.queue_family_index;
      }
    };

    /*
      Factory
    */

    class Factory final {
     public:
      explicit Factory(VkDevice device);

      typedef Pool::Descriptor Descriptor;
      typedef VK_DELETER(CommandPool) Deleter;
      typedef Handle<VkCommandPool, Deleter> Handle;

      struct Hasher {
        inline size_t operator()(const Descriptor& descriptor) const {
          return c10::get_hash(descriptor.queue_family_index);
        }
      };

      Handle operator()(const Descriptor& descriptor) const;

     private:
      VkDevice device_;
    };

    /*
      Cache
    */

    typedef api::Cache<Factory> Cache;
    Cache cache;

    explicit Pool(const VkDevice device)
      : cache(Factory(device)) {
    }
  } pool;

  //
  // Buffer
  //

  class Buffer final {
   public:
    Buffer(VkDevice device, VkCommandPool command_pool);

    void begin();
    void end();

    void bind(VkPipeline pipeline);
    void bind(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptor_set);
    void dispatch();

   private:
    VkCommandBuffer command_buffer_;
  };

  explicit Command(const VkDevice device)
    : pool(device) {
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
