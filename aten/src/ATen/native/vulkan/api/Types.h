#pragma once

#ifdef USE_VULKAN_API
namespace at {
namespace native {
namespace vulkan {
namespace api {

enum class StorageType {
  BUFFER,
  TEXTURE_3D,
  TEXTURE_2D,
  UNKNOWN,
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
