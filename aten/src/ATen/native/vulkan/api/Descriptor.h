#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Descriptor final {
  /*
    Cache
  */

  class Cache final {
   public:
    Cache(VkDevice device, VkDescriptorPool descriptor_pool);

    VkDescriptorSet allocate(VkDescriptorSetLayout descriptor_set_layout);
    void purge();

   private:
    VkDevice device_;
    VkDescriptorPool descriptor_pool_;
  };

  //
  // Pool
  //

  struct Pool final {
    /*
      Descriptor
    */

    struct Descriptor final {
      struct Size final {
        VkDescriptorType type;
        uint32_t count;

        inline bool operator==(const Size& size) const {
          return (type == size.type) && (count == size.count);
        }

        inline size_t hash() const {
          return c10::get_hash(type, count);
        }
      };

      uint32_t capacity;
      std::array<Size, 16u> sizes;
    };

    /*
      Factory
    */

    class Factory final {
     public:
      explicit Factory(VkDevice device);

      typedef Pool::Descriptor Descriptor;
      typedef VK_DELETER(DescriptorPool) Deleter;
      typedef Handle<VkDescriptorPool, Deleter> Handle;

      struct Hasher {
        inline size_t operator()(const Descriptor& descriptor) const {
          size_t hash = c10::get_hash(descriptor.capacity);

          for (const Descriptor::Size& size : descriptor.sizes) {
            hash = c10::hash_combine(hash, size.hash());
          }

          return hash;
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
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

