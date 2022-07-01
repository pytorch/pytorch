#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils{

void pack_staging_to_vtensor(api::VulkanBuffer& staging, vTensor& v_self) {
  api::Context* const context = api::context();

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();

  const api::utils::uvec3 extents = v_self.extents();
  const uint32_t plane = extents.data[0u] * extents.data[1u];

  const struct Block final {
    api::utils::uvec3 extents;
    uint32_t block;
    api::utils::uvec4 offset;
  } block {
    extents,
    4u * plane,
    {
      0u * plane,
      1u * plane,
      2u * plane,
      3u * plane,
    },
  };

  api::UniformParamsBuffer params(context, block);

  context->dispatch(
      command_buffer,
      { // shader layout
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(nchw_to_image),  // shader
      extents,  // global work group size
      adaptive_work_group_size(extents),  // local work group size
      // shader inputs
      v_self.image(
        command_buffer,
        api::PipelineStage::Compute, api::MemoryAccessType::WRITE),
      staging.package(),
      // params buffer
      params.buffer().package());

  command_pool.submit(context->gpu().queue, command_buffer);
}

void pack_vtensor_to_staging(
    vTensor& v_self, api::VulkanBuffer& staging, const VkFence fence) {
  api::Context* const context = api::context();

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();

  const api::utils::uvec3 extents = v_self.extents();
  const uint32_t plane = extents.data[0u] * extents.data[1u];

  const struct Block final {
    api::utils::uvec3 extents;
    uint32_t block;
    api::utils::uvec4 offset;
  } block {
    extents,
    4u * plane,
    {
      0u * plane,
      1u * plane,
      2u * plane,
      3u * plane,
    },
  };

  api::UniformParamsBuffer params(context, block);

  context->dispatch(
      command_buffer,
      { // shader layout
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(image_to_nchw),  // shader
      extents,  // global work group size
      adaptive_work_group_size(extents),  // local work group size
      // shader inputs
      v_self.image(
        command_buffer,
        api::PipelineStage::Compute),
      staging.package(),
      // params buffer
      params.buffer().package());

  command_pool.submit(context->gpu().queue, command_buffer, fence);
}


} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
