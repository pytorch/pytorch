#include <ATen/native/vulkan/api/Utils.h>

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {

std::string stringize(const VkExtent3D& extents) {
  std::stringstream ss;
  ss << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return ss.str();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
