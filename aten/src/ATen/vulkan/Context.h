#ifdef USE_VULKAN
#include <ATen/native/vulkan/VulkanAten.h>

namespace at {
namespace vulkan {
inline bool is_available() {
  return at::native::is_vulkan_available();
}
} // namespace vulkan
} // namespace at
#else
namespace at {
namespace vulkan {
inline bool is_available() {
  return false;
}
} // namespace vulkan
} // namespace at
#endif
