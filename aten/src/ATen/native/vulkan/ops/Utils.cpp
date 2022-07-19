#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

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
  } block{
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
  bool is_quantized = v_self.is_quantized();
  api::ShaderSource kernel = is_quantized ? VK_KERNEL(nchw_to_image_quantized)
                                          : VK_KERNEL(nchw_to_image);

  context->submit_compute_job(
      // shader descriptor
      kernel,
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
    vTensor& v_self,
    api::VulkanBuffer& staging,
    const VkFence fence_handle) {
  api::Context* const context = api::context();

  const api::utils::uvec3 extents = v_self.extents();
  const uint32_t plane = extents.data[0u] * extents.data[1u];

  const struct Block final {
    api::utils::uvec3 extents;
    uint32_t block;
    api::utils::uvec4 offset;
  } block{
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
  bool is_quantized = v_self.is_quantized();
  api::utils::uvec3 copy_extents;
  copy_extents.data[0u] = 1;
  copy_extents.data[1u] = 1;
  copy_extents.data[2u] =
      ((v_self.sizes()[1] * v_self.sizes()[2] * v_self.sizes()[3]) / 4);
  api::ShaderSource kernel = is_quantized ? VK_KERNEL(image_to_nchw_quantized)
                                          : VK_KERNEL(image_to_nchw);
  api::utils::uvec3 extents_to_use = is_quantized ? copy_extents : extents;

  context->submit_compute_job(
      // shader descriptor
      kernel,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      extents_to_use,
      // local work group size
      adaptive_work_group_size(extents_to_use),
      // fence handle
      fence_handle,
      // shader arguments
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      staging,
      // params buffer
      params.buffer());
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
