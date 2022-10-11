#pragma once

#ifdef USE_VULKAN_API
namespace at {
namespace native {
namespace vulkan {
namespace api {

enum class StorageType {
  Buffer,
  Image,
  Unknown
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
