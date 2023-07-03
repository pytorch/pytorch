#include <ATen/native/vulkan/api/api.h>

namespace at {
namespace native {
namespace vulkan {
namespace packing {

api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst);
api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src);

void record_nchw_to_image_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle);

void record_image_to_nchw_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle);

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle);

void record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle);

} // namespace packing
} // namespace vulkan
} // namespace native
} // namespace at
