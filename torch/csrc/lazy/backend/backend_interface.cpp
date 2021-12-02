#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch {
namespace lazy {

namespace {
std::atomic<const BackendImplInterface*> backend_impl_registry;
} // namespace

BackendRegistrar::BackendRegistrar(
    const BackendImplInterface* backend_impl_interface) {
  backend_impl_registry.store(backend_impl_interface);
}

const BackendImplInterface* getBackend() {
  auto* interface = backend_impl_registry.load();
  TORCH_CHECK(interface, "Lazy tensor backend not registered.");
  return interface;
}

}  // namespace lazy
}  // namespace torch
