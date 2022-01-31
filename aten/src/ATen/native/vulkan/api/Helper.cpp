#include <ATen/native/vulkan/api/Helper.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace helper {

#ifdef USE_VULKAN_API

void copy_texture_to_texture(
    api::Command::Buffer& command_buffer,
    api::Resource::Image::Object& src_image,
    api::Resource::Image::Object& dst_image,
    api::utils::uvec3 copy_extents,
    api::utils::uvec3 src_offset,
    api::utils::uvec3 dst_offset) {
  VkImageCopy copy_info{};
  copy_info.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_info.srcSubresource.layerCount = 1;
  copy_info.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_info.dstSubresource.layerCount = 1;
  copy_info.extent.width = copy_extents.data[0u];
  copy_info.extent.height = copy_extents.data[1u];
  copy_info.extent.depth = copy_extents.data[2u];
  copy_info.srcOffset.x = src_offset.data[0u];
  copy_info.srcOffset.y = src_offset.data[1u];
  copy_info.srcOffset.z = src_offset.data[2u];
  copy_info.dstOffset.x = dst_offset.data[0u];
  copy_info.dstOffset.y = dst_offset.data[1u];
  copy_info.dstOffset.z = dst_offset.data[2u];

  // To use vkCmdCopyImage, the stage of src & dst image must be set to vTensor::Stage::Transfer.
  vkCmdCopyImage(
    command_buffer.handle(),
    src_image.handle, VK_IMAGE_LAYOUT_GENERAL,
    dst_image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1,
    &copy_info);
}

#endif /* USE_VULKAN_API */

} // namespace helper
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
