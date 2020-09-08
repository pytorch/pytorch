#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

Pipeline::Layout::Factory::Factory(const VkDevice device)
 : device_(device) {
  TORCH_INTERNAL_ASSERT(device_, "Invalid Vulkan device!");
}

typename Pipeline::Layout::Factory::Handle Pipeline::Layout::Factory::operator()(
    const Descriptor& descriptor) const {
  TORCH_INTERNAL_ASSERT(
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
      device_, &pipeline_layout_create_info, nullptr, &pipeline_layout));

  return Handle{
    pipeline_layout,
    Deleter(device_),
  };
}

namespace {

VkPipelineCache create_pipeline_cache(const VkDevice device) {
  TORCH_INTERNAL_ASSERT(device, "Invalid Vulkan device!");

  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    nullptr,
    0u,
    0u,
    nullptr,
  };

  VkPipelineCache pipeline_cache{};
  VK_CHECK(vkCreatePipelineCache(
      device, &pipeline_cache_create_info, nullptr, &pipeline_cache));

  return pipeline_cache;
}

} // namespace

Pipeline::Factory::Factory(const VkDevice device)
 : device_(device),
   pipeline_cache_(
      create_pipeline_cache(device),
      VK_DELETER(PipelineCache)(device)) {
}

typename Pipeline::Factory::Handle Pipeline::Factory::operator()(
    const Descriptor& descriptor) const {
  TORCH_INTERNAL_ASSERT(
      descriptor.pipeline_layout,
      "Invalid Vulkan pipeline layout!");

  TORCH_INTERNAL_ASSERT(
      descriptor.shader_module,
      "Invalid Vulkan shader module!");

  constexpr uint32_t x_offset = 0u;
  constexpr uint32_t x_size = sizeof(Shader::WorkGroup::x);
  constexpr uint32_t y_offset = x_offset + x_size;
  constexpr uint32_t y_size = sizeof(Shader::WorkGroup::y);
  constexpr uint32_t z_offset = y_offset + y_size;
  constexpr uint32_t z_size = sizeof(Shader::WorkGroup::z);

  constexpr VkSpecializationMapEntry specialization_map_entires[3]{
    // X
    {
      1u,
      x_offset,
      x_size,
    },
    // Y
    {
      2u,
      y_offset,
      y_size,
    },
    // Z
    {
      3u,
      z_offset,
      z_size,
    },
  };

  const VkSpecializationInfo specialization_info{
    3u,
    specialization_map_entires,
    sizeof(Shader::WorkGroup),
    &descriptor.work_group,
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

  return Handle{
    pipeline,
    Deleter(device_),
  };
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
