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
// from those pools, intended to minimize redundant object reconstructions or
// accelerate unavoidable memory allocations, both at the cost of extra memory
// consumption.
//
// A descriptor set is logically an array of descriptors, each of which
// references a resource (i.e. buffers and images), in turn telling the core
// executing the shader, where in GPU, or GPU-accessible system, memory the said
// resource resides.
//
// To accelerate creation of the descriptor sets, modern graphics APIs allocate
// them from a pool, more elaborately referred to as descriptor pools, which do
// need to be purged frequently _after_ none of the descriptors the pools contain
// is in use by the GPU.  Care must be taken that descriptors are not freed while
// they are in use by the pipeline, which considering the asynchronous nature of
// CPU-GPU interactions, can be anytime after the command is issued until it is
// fully executed by the GPU.
//
// As you can imagine, it is possible to have multiple descriptor pools, each of
// which is configured to house different types of descriptor sets with different
// allocation strategies. These descriptor pools themselves are fairly stable
// objects in that they theymself should not be created and destroyed frequently.
// That is the reason why we store them in a cache, which according to our usage
// of the term 'cache' in this implementatoin, is reserved for objects that are
// created infrequently and stabilize to a manageable number quickly over the
// lifetime of the program.
//
// Descriptor sets though, on the other hand, are allocated from pools which
// indeed does mean that the pools must be purged on a regular basis or else
// they will run out of free items.  Again, this is in line with our usage of
// the term 'pool' in this implementation which we use to refer to a container
// of objects that is allocated out of and is required to be frequently purged.
//
// It is important to point out that for performance reasons, we intentionally
// do not free the descriptor sets individually, and instead opt to purge the
// pool in its totality, even though Vulkan supports the former usage pattern
// as well.  This behavior is by design.
//

struct C10_EXPORT Descriptor final {
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

        bool operator==(const Size& size) const;
        size_t hash() const;
      };

      uint32_t capacity;
      std::array<Size, 16u> sizes;

      bool operator==(const Descriptor& descriptor) const;
    };

    static constexpr Descriptor kDefault{
      1024u,
      {
        // Note: It is OK for the sum of descriptors per type, below, to exceed
        // the max total figure above, but be concenious of memory consumption.
        // Considering how the descriptor pool must be frequently purged anyway
        // as a result of the impracticality of having enormous pools that
        // persist through the execution of the program, there is diminishing
        // return in increasing max counts.
        {
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

    explicit Pool(const VkDevice device)
      : cache(Factory(device)) {
    }

    static void purge(VkDevice device, VkDescriptorPool descriptor_pool);
  } pool;

  /*
    Factory
  */

  class Factory final {
   public:
    Factory(VkDevice device, VkDescriptorPool descriptor_pool);

    VkDescriptorSet allocate(VkDescriptorSetLayout descriptor_set_layout);
    void purge();

   private:
    VkDevice device_;
    VkDescriptorPool descriptor_pool_;
  } factory;

  explicit Descriptor(const VkDevice device)
    : pool(device),
      factory(device, pool.cache.retrieve(Pool::kDefault)) {
  }
};

//
// Impl
//

inline bool Descriptor::Pool::Descriptor::Size::operator==(
    const Size& size) const {
  return (type == size.type) &&
         (count == size.count);
}

inline size_t Descriptor::Pool::Descriptor::Size::hash() const {
  return c10::get_hash(type, count);
}

inline bool Descriptor::Pool::Descriptor::operator==(
    const Descriptor& descriptor) const {
  return (capacity == descriptor.capacity) &&
         (sizes == descriptor.sizes);
}

inline size_t Descriptor::Pool::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  size_t hash = c10::get_hash(descriptor.capacity);

  for (const Descriptor::Size& size : descriptor.sizes) {
    hash = c10::hash_combine(hash, size.hash());
  }

  return hash;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
