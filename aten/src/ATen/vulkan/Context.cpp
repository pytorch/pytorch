#include <atomic>

#include <ATen/Tensor.h>
#include <ATen/vulkan/Context.h>

#ifdef USE_VULKAN_API
#include <ATen/native/vulkan/api/Context.h>
#endif /* USE_VULKAN_API */

namespace at {
namespace vulkan {

std::atomic<const VulkanImplInterface*> g_vulkan_impl_registry;

VulkanImplRegistrar::VulkanImplRegistrar(VulkanImplInterface* impl) {
  g_vulkan_impl_registry.store(impl);
}

at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src) {
  auto p = at::vulkan::g_vulkan_impl_registry.load();
  if (p) {
    return p->vulkan_copy_(self, src);
  }
  AT_ERROR("Vulkan backend was not linked to the build");
}
} // namespace vulkan

namespace native {
bool is_vulkan_available() {
#ifdef USE_VULKAN_API
  return native::vulkan::api::available();
#else
  auto p = at::vulkan::g_vulkan_impl_registry.load();
  return p ? p->is_vulkan_available() : false;
#endif
}
} // namespace native

} // namespace at
