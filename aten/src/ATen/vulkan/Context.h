#include "ATen/native/vulkan/VulkanAten.h"

namespace at {
namespace vulkan {
inline bool is_available() {
  return at::native::_vulkan_available();
}
} // namespace vulkan
} // namespace at
