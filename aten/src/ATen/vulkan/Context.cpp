#include <ATen/Tensor.h>
#include <ATen/native/vulkan/api/Context.h>

namespace at {
namespace native {

bool is_vulkan_available() {
  return native::vulkan::api::available();
}

} // namespace native
} // namespace at
