#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch {
namespace lazy {

namespace {
std::atomic<const BackendImplInterface*> backend_impl_registry;
} // namespace

bool hasBackend() {
  return !!backend_impl_registry.load();
}

const BackendImplInterface* getBackend() {
  auto* interface = backend_impl_registry.load();
  TORCH_CHECK(interface, "Lazy tensor backend not registered.");
  return interface;
}

BackendRegistrar::BackendRegistrar(
    const BackendImplInterface* backend_impl_interface) {
  backend_impl_registry.store(backend_impl_interface);
}

at::Tensor MakeTensorFromComputationData(
    const BackendDataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type) {
  return getBackend()->MakeTensorFromComputationData(data, logical_scalar_type);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<Node*> post_order,
    Util::EmissionMap emit_status) {
  return getBackend()->CreateLoweringContext(
      name, device, post_order, emit_status);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name,
    BackendDevice device) {
  return getBackend()->CreateLoweringContext(name, device);
}

}  // namespace lazy
}  // namespace torch
