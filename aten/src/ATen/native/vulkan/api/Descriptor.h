#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Descriptor final {
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
          return (type == size.type) &&
                 (count == size.count);
        }

        inline size_t hash() const {
          return c10::get_hash(type, count);
        }
      };

      uint32_t capacity;
      std::array<Size, 16u> sizes;

      inline bool operator==(const Descriptor& descriptor) const {
        return (capacity == descriptor.capacity) &&
               (sizes == descriptor.sizes);
      }
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
    Cache cache;

    // Default Configuration
    static constexpr Descriptor kDefault{
      1024u,
      {
        {
          // Note: It is OK for the sum of descriptors per type, below, to
          // exceed the max total figure above, but be concenious of memory
          // consumption.  The cache must be purged frequently anyway so
          // having enormous descriptor caches that will persist through the
          // entirety of the application's lifetime is not practical.

          /*
            Buffers
          */

          {
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            256u,
          },
          {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            256u,
          },

          /*
            Images
          */

          {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            256u,
          },
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            256u,
          },
        },
      },
    };

    explicit Pool(const VkDevice device)
      : cache(Factory(device)) {
    }

  } pool;

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
  } cache;

  explicit Descriptor(const VkDevice device)
    : pool(device),
      cache(device, pool.cache.retrieve(Pool::kDefault)) {
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

