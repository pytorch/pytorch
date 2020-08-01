#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Pipeline final {
  struct Descriptor final {
    VkShaderModule shader_module;
    VkPipelineLayout pipeline_layout;
  };

  class Factory final {
   public:
    explicit Factory(VkDevice device);

    typedef Pipeline::Descriptor Descriptor;
    typedef VK_DELETER(Pipeline) Deleter;
    typedef Handle<VkPipeline, Deleter> Handle;

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    api::Handle<VkPipelineCache, VK_DELETER(PipelineCache)> pipeline_cache_;
  };

  class Cache final {
   public:
    explicit Cache(VkDevice device);
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = default;
    Cache& operator=(Cache&&) = default;
    ~Cache() = default;

    VkPipeline retrieve(const Descriptor& descriptor);

   private:
    api::Cache<Descriptor, VkPipeline, Factory> cache_;
  };

  struct Layout final {
    struct Descriptor final {
      VkDescriptorSetLayout descriptor_set_layout;
    };

    class Factory final {
     public:
      explicit Factory(VkDevice device);

      typedef Layout::Descriptor Descriptor;
      typedef VK_DELETER(PipelineLayout) Deleter;
      typedef Handle<VkPipelineLayout, Deleter> Handle;

      Handle operator()(const Descriptor& descriptor) const;

     private:
      VkDevice device_;
    };

    class Cache final {
     public:
      explicit Cache(VkDevice device);
      Cache(const Cache&) = delete;
      Cache& operator=(const Cache&) = delete;
      Cache(Cache&&) = default;
      Cache& operator=(Cache&&) = default;
      ~Cache() = default;

      VkPipelineLayout retrieve(const Descriptor& descriptor);

     private:
      api::Cache<Descriptor, VkPipelineLayout, Factory> cache_;
    };
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
