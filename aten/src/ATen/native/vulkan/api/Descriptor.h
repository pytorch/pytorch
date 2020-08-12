#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// This struct defines caches of descriptor pools, and descriptor sets allocated
// from those pools intended to minimize redundant object reconstructions at the
// cost of extra memory consumption.
//
// A desciprotr set is effectively a table of pointers / handles that tells the
// shader where in (CPU or GPU) memory the resources (i.e. buffers and images)
// it interacts with reside.  These tables, henceforth referred to as descriptor
// sets as Vulkan calls them, are re-created continuously in a frame as the
// resources they point to move around in memory frequently from one invocation
// to another.  To accelerate creation of the descriptor sets, modern graphics
// APIs allocate them from a pool, more elaborately referred to as descriptor
// pools, which indeed do need to be purged frequently _after_ none of the
// descriptors the pools contain is in use by the GPU.  Care must be taken
// that descriptors are not freed while they are in use by the pipeline.
//
// As you can imagine, it is possible to have multiple descriptor pools,
// each of which configured to house different types of descriptor sets with
// different allocation schemes. These descriptor pools themselves are fairly
// stable objects in that they theymself should not be created and destroyed
// frequently.  That is the reason why we store them in a cache, which according
// to our usage of the term cache here, is reserved for objects that are created
// infrequently and stabilize to a very manageable number quickly over the
// lifetime of the program.
//
// Descriptor sets though, on the other hand, are allocated from pools
// frequently, which indeed does mean that the pools must be purged on a regular
// basis or else they will run out of free items.  Again, this is in line with
// our usage of the term pool which we use here to refer to a container for
// objects that are required to be purged once per frame.
//
// It is important to point out that for performance reasons, we intentionally do
// not free the descriptor sets individually, and instead purge the cache in its
// totality once per frame, even though Vulkan supports the former usage pattern
// as well.  This is by design.
//

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

