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
        size_t operator()(const Descriptor& descriptor) const;
      };

      Handle operator()(const Descriptor& descriptor) const;
      void purge(VkCommandPool command_pool);

     private:
      VkDevice device_;
    };

    /*
      Cache
    */

    typedef api::Cache<Factory> Cache;
    Cache cache;

    // This field simply stores a reference to the primary command pool in
    // the cache for ease of access, and carries no significance otherwise.
    // This object's lifetime is managed by the cache as usual.  Purge the
    // contents of the pool regularly through the factory it was created.

    VkCommandPool primary;

    Pool(VkDevice device, const Descriptor& primary);
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

  explicit Command(const VkDevice device, const Pool::Descriptor& primary)
    : pool(device, primary) {
  }
};

//
// Impl
//

inline bool operator==(
    const Command::Pool::Descriptor& _1,
    const Command::Pool::Descriptor& _2) {
  return _1.queue_family_index == _2.queue_family_index;
}

inline size_t Command::Pool::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  return c10::get_hash(descriptor.queue_family_index);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
