#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Pipeline final {
  struct Descriptor final {
    VkShaderModule shader_module;
    VkPipelineLayout pipeline_layout;

    inline bool operator==(const Descriptor& descriptor) const {
      return (shader_module == descriptor.shader_module) &&
             (pipeline_layout == descriptor.pipeline_layout);
    }
  };

  class Factory final {
   public:
    explicit Factory(VkDevice device);

    typedef Pipeline::Descriptor Descriptor;
    typedef VK_DELETER(Pipeline) Deleter;
    typedef Handle<VkPipeline, Deleter> Handle;

    struct Hasher {
      inline size_t operator()(const Descriptor& descriptor) const {
        return c10::get_hash(
            descriptor.shader_module,
            descriptor.pipeline_layout);
      }
    };

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    api::Handle<VkPipelineCache, VK_DELETER(PipelineCache)> pipeline_cache_;
  };

  typedef api::Cache<Factory> Cache;

  struct Layout final {
    struct Descriptor final {
      VkDescriptorSetLayout descriptor_set_layout;

      inline bool operator==(const Descriptor& descriptor) const{
        return (descriptor_set_layout == descriptor.descriptor_set_layout);
      }
    };

    class Factory final {
     public:
      explicit Factory(VkDevice device);

      typedef Layout::Descriptor Descriptor;
      typedef VK_DELETER(PipelineLayout) Deleter;
      typedef Handle<VkPipelineLayout, Deleter> Handle;

      struct Hasher {
        inline size_t operator()(const Descriptor& descriptor) const {
          return c10::get_hash(descriptor.descriptor_set_layout);
        }
      };

      Handle operator()(const Descriptor& descriptor) const;

     private:
      VkDevice device_;
    };

    typedef api::Cache<Factory> Cache;
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

