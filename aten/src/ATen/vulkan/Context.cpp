#include <ATen/Tensor.h>
#include <ATen/native/vulkan/api/Context.h>

namespace at {
namespace native {

bool is_vulkan_available() {
#ifdef USE_VULKAN
  return native::vulkan::api::available();
#else
  return false;
#endif /* USE_VULKAN
}

} // namespace native
} // namespace at
