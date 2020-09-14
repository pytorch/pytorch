#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Adapter.h>
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
    Buffer(VkDevice device, VkCommandPool command_pool);

    void begin();
    void end();

    void bind(VkPipeline pipeline);
    void bind(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptor_set);
    void copy(VkBuffer source, VkBuffer destination, size_t size);
    void dispatch();

    void submit(VkQueue queue, VkFence fence);

   private:
    VkCommandBuffer command_buffer_;
  };

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
      explicit Factory(const GPU& gpu);

      typedef Pool::Descriptor Descriptor;
      typedef VK_DELETER(CommandPool) Deleter;
      typedef Handle<VkCommandPool, Deleter> Handle;

      struct Hasher {
        size_t operator()(const Descriptor& descriptor) const;
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

    /*
      Object
    */

    class Object final {
     public:
      Object(VkDevice device, VkCommandPool command_pool);

      Buffer allocate();
      void purge();

     private:
      VkDevice device_;
      VkCommandPool command_pool_;
    } primary /* [thread_count] */;

    explicit Pool(const GPU& gpu);
  } pool;

  explicit Command(const GPU& gpu)
    : pool(gpu) {
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
