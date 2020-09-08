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

struct Descriptor final {
  //
  // Pool
  //

  struct Pool final {
    /*
      Descriptor
    */

    struct Descriptor final {
      uint32_t capacity;
      c10::SmallVector<VkDescriptorPoolSize, 16u> sizes;
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
      void purge(VkDescriptorPool descriptor_pool);

     private:
      VkDevice device_;
    };

    /*
      Cache
    */

    typedef api::Cache<Factory> Cache;
    Cache cache;

    // This field simply stores a reference to the primary descriptor pool in
    // the cache for ease of access, and carries no significance otherwise.
    // This object's lifetime is managed by the cache as usual.  Purge the
    // contents of the pool regularly through the factory it was created.

    VkDescriptorPool primary;

    explicit Pool(VkDevice device);
  } pool;

  /*
    Set
  */

  class Set final {
   public:
    Set(VkDevice device, VkDescriptorPool descriptor_pool);

    VkDescriptorSet allocate(VkDescriptorSetLayout descriptor_set_layout);

   private:
    VkDevice device_;
    VkDescriptorPool descriptor_pool_;
  } set;

  explicit Descriptor(const VkDevice device)
    : pool(device),
      set(device, pool.primary) {
  }
};

//
// Impl
//

inline bool operator==(
    const Descriptor::Pool::Descriptor& _1,
    const Descriptor::Pool::Descriptor& _2) {
  return (_1.capacity == _2.capacity) &&
         (_1.sizes == _2.sizes);
}

inline size_t Descriptor::Pool::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  size_t hash = c10::get_hash(descriptor.capacity);

  for (const VkDescriptorPoolSize& descriptor_pool_size : descriptor.sizes) {
    hash = c10::hash_combine(
        hash,
        c10::get_hash(
            descriptor_pool_size.type,
            descriptor_pool_size.descriptorCount));
  }

  return hash;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

inline bool operator==(
    const VkDescriptorPoolSize& descriptor_pool_size_1,
    const VkDescriptorPoolSize& descriptor_pool_size_2) {
  return (descriptor_pool_size_1.type == descriptor_pool_size_2.type) &&
         (descriptor_pool_size_1.descriptorCount == descriptor_pool_size_2.descriptorCount);
}
