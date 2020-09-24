#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
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
      explicit Factory(VkDevice device);

      typedef Layout::Descriptor Descriptor;
      typedef VK_DELETER(PipelineLayout) Deleter;
      typedef Handle<VkPipelineLayout, Deleter> Handle;

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

    explicit Layout(const VkDevice device)
      : cache(Factory(device)) {
    }
  } layout;

  /*
    Descriptor
  */

  struct Descriptor final {
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    Shader::WorkGroup work_group;
  };

  /*
    Factory
  */

  class Factory final {
   public:
    explicit Factory(VkDevice device);

    typedef Pipeline::Descriptor Descriptor;
    typedef VK_DELETER(Pipeline) Deleter;
    typedef Handle<VkPipeline, Deleter> Handle;

    struct Hasher {
      size_t operator()(const Descriptor& descriptor) const;
    };

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    api::Handle<VkPipelineCache, VK_DELETER(PipelineCache)> pipeline_cache_;
  };

  /*
    Cache
  */

  typedef api::Cache<Factory> Cache;
  Cache cache;

  explicit Pipeline(const VkDevice device)
    : layout(device),
      cache(Factory(device)) {
  }
};

//
// Impl
//

inline bool operator==(
    const Pipeline::Layout::Descriptor& _1,
    const Pipeline::Layout::Descriptor& _2) {
  return (_1.descriptor_set_layout == _2.descriptor_set_layout);
}

inline size_t Pipeline::Layout::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  return c10::get_hash(descriptor.descriptor_set_layout);
}

inline bool operator==(
    const Pipeline::Descriptor& _1,
    const Pipeline::Descriptor& _2) {
  return (_1.pipeline_layout == _2.pipeline_layout) &&
         (_1.shader_module == _2.shader_module) &&
         (_1.work_group == _2.work_group);
}

inline size_t Pipeline::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  return c10::get_hash(
      descriptor.pipeline_layout,
      descriptor.shader_module,
      descriptor.work_group.x,
      descriptor.work_group.y,
      descriptor.work_group.z);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
