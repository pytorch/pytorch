#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

namespace torch_lazy_tensors {
namespace compiler {

std::atomic<const BackendImplInterface*> backend_impl_registry;

BackendRegistrar::BackendRegistrar(
    const BackendImplInterface* backend_impl_interface) {
  backend_impl_registry.store(backend_impl_interface);
}

std::vector<std::string> GetCompilationDevices(
    const std::string& device, c10::ArrayRef<std::string> devices) {
  return compiler::getBackend()
      ->GetCompilationDevices(device, devices);
}

at::Tensor MakeTensorFromComputationData(
    const torch::lazy::BackendDataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type) {
  return compiler::getBackend()
      ->MakeTensorFromComputationData(data, logical_scalar_type);
}

}  // namespace compiler

namespace ir {

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, torch::lazy::BackendDevice device,
    c10::ArrayRef<torch::lazy::Node*> post_order,
    torch::lazy::Util::EmissionMap emit_status) {
  return compiler::getBackend()
      ->CreateLoweringContext(name, device, post_order, emit_status);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, torch::lazy::BackendDevice device) {
  return compiler::getBackend()
      ->CreateLoweringContext(name, device);
}

}  // namespace ir
}  // namespace torch_lazy_tensors
