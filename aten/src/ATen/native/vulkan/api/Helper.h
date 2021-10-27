#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Resource.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace helper {
//
// Copy Texture
//

void copy_texture_to_texture(
    api::Command::Buffer& command_buffer,
    api::Resource::Image::Object& src_image,
    api::Resource::Image::Object& dst_image,
    api::utils::uvec3 copy_extents,
    api::utils::uvec3 src_offset,
    api::utils::uvec3 dst_offset);

} // namespace utils
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
