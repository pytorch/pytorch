#include <atomic>

#include <ATen/Tensor.h>
#include <ATen/metal/Context.h>

namespace at::metal {

std::atomic<const MetalInterface*> g_metal_impl_registry;

MetalImplRegistrar::MetalImplRegistrar(MetalInterface* impl) {
  g_metal_impl_registry.store(impl);
}

at::Tensor& metal_copy_(at::Tensor& self, const at::Tensor& src) {
  auto p = at::metal::g_metal_impl_registry.load();
  if (p) {
    return p->metal_copy_(self, src);
  }
  AT_ERROR("Metal backend was not linked to the build");
}
} // namespace at::metal

namespace at::native {
bool is_metal_available() {
  auto p = at::metal::g_metal_impl_registry.load();
  return p ? p->is_metal_available() : false;
}

} // namespace at::native
