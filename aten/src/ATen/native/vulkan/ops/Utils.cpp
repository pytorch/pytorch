#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils{

void pack_buffer_to_vtensor(
    api::VulkanBuffer& buffer,
    vTensor& v_self,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

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

  context->submit_compute_job(
      // shader layout signature
      {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      // shader descriptor
      VK_KERNEL(nchw_to_image),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      extents,
      // local work group size
      adaptive_work_group_size(extents),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      buffer,
      // params buffer
      params.buffer());
}

void pack_staging_to_vtensor(api::VulkanBuffer& staging, vTensor& v_self) {
  api::PipelineBarrier pipeline_barrier{};
  pack_buffer_to_vtensor(staging, v_self, pipeline_barrier);
}

void pack_vtensor_to_staging(
    vTensor& v_self, api::VulkanBuffer& staging, const VkFence fence_handle) {
  api::Context* const context = api::context();

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
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader layout signature
      {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      // shader descriptor
      VK_KERNEL(image_to_nchw),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      extents,
      // local work group size
      adaptive_work_group_size(extents),
      // fence handle
      fence_handle,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE),
      staging,
      // params buffer
      params.buffer());
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
