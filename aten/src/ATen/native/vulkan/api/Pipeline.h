#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// This struct defines pipeline, and pipeline layout, caches intended to minimize
// redundant object reconstructions at the cost of extra memory consumption.
//
// A Vulkan pipeline contains the entirety of states, as one coherent monolithic
// bundle, required to configure the GPU's execution pipeline.  This usage
// pattern minimizes driver overhead, promotes pipeline state reuse, and is a
// departure from, and in direct contrast with, OpenGL's individually confiurable
// state machine.
//
// A Vulkan pipeline layout represents a sequence of Vulkan descriptor sets each
// having a specific layout, and deterimines the interface between all shader
// stages and shader resources.  For more information on shaders and shader
// layouts check the description of at::navie::vulkan::api::Shader.
//
// This struct defines the facilities required to create, reuse, and destruct
// these Vulkan objects.
//

struct Pipeline final {
  //
  // Barrier
  //

  struct Barrier final {
    struct Stage final {
      VkPipelineStageFlags src;
      VkPipelineStageFlags dst;
    } stage;

    c10::SmallVector<Resource::Buffer::Barrier, 4u> buffers;
    c10::SmallVector<Resource::Image::Barrier, 4u> images;

    operator bool() const;
  };

  //
  // Layout
  //

  struct Layout final {
    /*
      Descriptor
    */

    struct Descriptor final {
      VkDescriptorSetLayout descriptor_set_layout;
    };

    /*
      Factory
    */

    class Factory final {
     public:
      explicit Factory(const GPU& gpu);

      typedef Layout::Descriptor Descriptor;
      typedef VK_DELETER(PipelineLayout) Deleter;
      typedef api::Handle<VkPipelineLayout, Deleter> Handle;

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

    explicit Layout(const GPU& gpu)
      : cache(Factory(gpu)) {
    }
  } layout;

  //
  // Stage
  //

  struct Stage final {
    typedef uint8_t Flags;

    enum Type : Flags {
      None = 0u << 0u,
      Compute = 1u << 0u,
      Host = 1u << 1u,
      Transfer = 1u << 2u,
    };
  };

  /*
    Descriptor
  */

  struct Descriptor final {
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    Shader::WorkGroup local_work_group;
  };

  /*
    Factory
  */

  class Factory final {
   public:
    explicit Factory(const GPU& gpu);

    typedef Pipeline::Descriptor Descriptor;
    typedef VK_DELETER(Pipeline) Deleter;
    typedef api::Handle<VkPipeline, Deleter> Handle;

    struct Hasher {
      size_t operator()(const Descriptor& descriptor) const;
    };

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    api::Handle<VkPipelineCache, VK_DELETER(PipelineCache)> pipeline_cache_;
  };

  /*
    Object
  */

  struct Object final {
    VkPipeline handle;
    VkPipelineLayout layout;
    Shader::WorkGroup local_work_group;

    operator bool() const;
  };

  /*
    Cache
  */

  class Cache final {
   public:
    explicit Cache(Factory factory);
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = default;
    Cache& operator=(Cache&&) = default;
    ~Cache() = default;

    Object retrieve(const Descriptor& descriptor);
    void purge();

   private:
    api::Cache<Factory> cache_;
  } cache;

  explicit Pipeline(const GPU& gpu)
    : layout(gpu),
      cache(Factory(gpu)) {
  }
};

//
// Impl
//

inline Pipeline::Barrier::operator bool() const {
  return (0u != stage.src) ||
         (0u != stage.dst) ||
         !buffers.empty() ||
         !images.empty();
}

inline bool operator==(
    const Pipeline::Layout::Descriptor& _1,
    const Pipeline::Layout::Descriptor& _2) {
  static_assert(
      std::is_trivially_copyable<Pipeline::Layout::Descriptor>::value,
      "This implementation is no longer valid!");

  return (0 == memcmp(&_1, &_2, sizeof(Pipeline::Layout::Descriptor)));
}

inline size_t Pipeline::Layout::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  return c10::get_hash(descriptor.descriptor_set_layout);
}

inline bool operator==(
    const Pipeline::Descriptor& _1,
    const Pipeline::Descriptor& _2) {
  static_assert(
      std::is_trivially_copyable<Pipeline::Descriptor>::value,
      "This implementation is no longer valid!");

  return (0 == memcmp(&_1, &_2, sizeof(Pipeline::Descriptor)));
}

inline size_t Pipeline::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  return c10::get_hash(
      descriptor.pipeline_layout,
      descriptor.shader_module,
      descriptor.local_work_group.data[0u],
      descriptor.local_work_group.data[1u],
      descriptor.local_work_group.data[2u]);
}

inline Pipeline::Object::operator bool() const {
  return (VK_NULL_HANDLE != handle) &&
         (VK_NULL_HANDLE != layout);
}

inline Pipeline::Object Pipeline::Cache::retrieve(
    const Descriptor& descriptor) {
  return {
    cache_.retrieve(descriptor),
    descriptor.pipeline_layout,
    descriptor.local_work_group,
  };
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
