#ifndef USE_VULKAN

namespace at {
namespace native {

bool is_vulkan_available() {
  return false;
}

} // namespace native
} // namespace at
#endif
