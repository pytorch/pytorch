#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VkPipelineCache create_pipeline_cache(const VkDevice device) {
  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    nullptr,
    0u,
    0u,
    nullptr,
  };

  VkPipelineCache pipeline_cache{};
  VK_CHECK(vkCreatePipelineCache(device, &pipeline_cache_create_info, nullptr, &pipeline_cache));

  return pipeline_cache;
}

} // namespace

Pipeline::Factory::Factory(const VkDevice device)
 : device_(device),
   pipeline_cache_(create_pipeline_cache(device), VK_DELETER(PipelineCache)(device)) {
}

typename Pipeline::Factory::Handle Pipeline::Factory::operator()(const Descriptor& descriptor) const {
  const VkSpecializationInfo specialization_info{
  };

  const VkComputePipelineCreateInfo compute_pipeline_create_info{
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    nullptr,
    0u,
    VkPipelineShaderStageCreateInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      nullptr,
      0u,
      VK_SHADER_STAGE_COMPUTE_BIT,
      descriptor.shader_module,
      "main",
      &specialization_info,
    },
    descriptor.pipeline_layout,
    VK_NULL_HANDLE,
    0u,
  };

  VkPipeline pipeline{};
  VK_CHECK(vkCreateComputePipelines(
    device_, pipeline_cache_.get(), 1u, &compute_pipeline_create_info, nullptr, &pipeline));

  return Handle{
    pipeline,
    Deleter(device_),
  };
}

Pipeline::Cache::Cache(const VkDevice device)
 : cache_(Factory(device)) {
}

VkPipeline Pipeline::Cache::retrieve(const Descriptor& descriptor) {
  return cache_.retrieve(descriptor, &descriptor);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
