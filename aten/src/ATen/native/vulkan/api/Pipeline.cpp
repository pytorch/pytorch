#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

Pipeline::Layout::Factory::Factory(const GPU& gpu)
 : device_(gpu.device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");
}

typename Pipeline::Layout::Factory::Handle Pipeline::Layout::Factory::operator()(
    const Descriptor& descriptor) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor.descriptor_set_layout,
      "Invalid Vulkan descriptor set layout!");

  const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    nullptr,
    0u,
    1u,
    &descriptor.descriptor_set_layout,
    0u,
    nullptr,
  };

  VkPipelineLayout pipeline_layout{};
  VK_CHECK(vkCreatePipelineLayout(
      device_,
      &pipeline_layout_create_info,
      nullptr,
      &pipeline_layout));

  TORCH_CHECK(
      pipeline_layout,
      "Invalid Vulkan pipeline layout!");

  return Handle{
    pipeline_layout,
    Deleter(device_),
  };
}

namespace {

VkPipelineCache create_pipeline_cache(const VkDevice device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    nullptr,
    0u,
    0u,
    nullptr,
  };

  VkPipelineCache pipeline_cache{};
  VK_CHECK(vkCreatePipelineCache(
      device,
      &pipeline_cache_create_info,
      nullptr,
      &pipeline_cache));

  TORCH_CHECK(
      pipeline_cache,
      "Invalid Vulkan pipeline cache!");

  return pipeline_cache;
}

} // namespace

Pipeline::Factory::Factory(const GPU& gpu)
 : device_(gpu.device),
   pipeline_cache_(
      create_pipeline_cache(device_),
      VK_DELETER(PipelineCache)(device_)) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      pipeline_cache_,
      "Invalid Vulkan pipeline cache!");
}

typename Pipeline::Factory::Handle Pipeline::Factory::operator()(
    const Descriptor& descriptor) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor.pipeline_layout,
      "Invalid Vulkan pipeline layout!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor.shader_module,
      "Invalid Vulkan shader module!");

  constexpr VkSpecializationMapEntry specialization_map_entires[3]{
    // X
    {
      0u,
      offsetof(Shader::WorkGroup, data[0u]),
      sizeof(Shader::WorkGroup::data[0u]),
    },
    // Y
    {
      1u,
      offsetof(Shader::WorkGroup, data[1u]),
      sizeof(Shader::WorkGroup::data[1u]),
    },
    // Z
    {
      2u,
      offsetof(Shader::WorkGroup, data[2u]),
      sizeof(Shader::WorkGroup::data[2u]),
    },
  };

  const VkSpecializationInfo specialization_info{
    3u,
    specialization_map_entires,
    sizeof(descriptor.local_work_group),
    &descriptor.local_work_group,
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
      device_,
      pipeline_cache_.get(),
      1u,
      &compute_pipeline_create_info,
      nullptr,
      &pipeline));

  TORCH_CHECK(
      pipeline,
      "Invalid Vulkan pipeline!");

  return Handle{
    pipeline,
    Deleter(device_),
  };
}

Pipeline::Cache::Cache(Factory factory)
  : cache_(std::move(factory)) {
}

void Pipeline::Cache::purge() {
  cache_.purge();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
