#include <ATen/native/vulkan/api/api.h>

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

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
    VkFence fence_handle);

bool record_image_to_nchw_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle);

vTensor convert_image_channels_packed_to_height_packed(const vTensor& v_input);

vTensor convert_image_channels_packed_to_width_packed(const vTensor& v_input);

} // namespace packing
} // namespace vulkan
} // namespace native
} // namespace at
