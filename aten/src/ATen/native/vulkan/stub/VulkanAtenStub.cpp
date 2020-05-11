#if !defined(USE_VULKAN) && !defined(USE_GLES)

namespace at {
namespace native {

bool is_vulkan_available() {
  return false;
}

} // namespace native
} // namespace at
#endif
