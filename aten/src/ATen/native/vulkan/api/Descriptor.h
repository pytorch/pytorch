#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>

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
  // Set
  //

  class Set final {
   public:
    Set(
        VkDevice device,
        VkDescriptorSet descriptor_set,
        const Shader::Layout::Signature& shader_layout_signature);
    Set(const Set&) = delete;
    Set& operator=(const Set&) = delete;
    Set(Set&&);
    Set& operator=(Set&&);
    ~Set() = default;

    Set& bind(uint32_t binding, const Resource::Buffer::Object& buffer);
    Set& bind(uint32_t binding, const Resource::Image::Object& image);

    VkDescriptorSet handle() const;

   private:
    void invalidate();

   private:
    struct Item final {
      uint32_t binding;
      VkDescriptorType type;

      union {
        VkDescriptorBufferInfo buffer;
        VkDescriptorImageInfo image;
      } info;
    };

    void update(const Item& item);

   private:
    VkDevice device_;
    VkDescriptorSet descriptor_set_;
    Shader::Layout::Signature shader_layout_signature_;

    struct {
      c10::SmallVector<Item, 6u> items;
      mutable bool dirty;
    } bindings_;
  };

  //
  // Pool
  //

  class Pool final {
   public:
    explicit Pool(const GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&);
    Pool& operator=(Pool&&);
    ~Pool();

    Set allocate(const Shader::Layout::Object& shader_layout);
    void purge();

   private:
    void invalidate();

   private:
    struct Configuration final {
      static constexpr uint32_t kQuantum = 16u;
      static constexpr uint32_t kReserve = 64u;
    };

    VkDevice device_;
    Handle<VkDescriptorPool, VK_DELETER(DescriptorPool)> descriptor_pool_;

    struct {
      struct Layout final {
        std::vector<VkDescriptorSet> pool;
        size_t in_use;
      };

      ska::flat_hash_map<VkDescriptorSetLayout, Layout> layouts;
    } set_;
  } pool /* [thread_count] */;

  explicit Descriptor(const GPU& gpu)
    : pool(gpu) {
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
