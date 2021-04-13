#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

namespace torch_lazy_tensors {
namespace compiler {

std::atomic<const BackendImplInterface*> backend_impl_registry;

BackendRegistrar::BackendRegistrar(
    BackendImplInterface* backend_impl_interface) {
  backend_impl_registry.store(backend_impl_interface);
}

std::unique_ptr<NodeLowering> NodeLowering::Create(ir::LoweringContext* loctx) {
  return getBackendRegistrar()->CreateNodeLowering(loctx);
}

NodeLowering* NodeLowering::Get() {
  return getBackendRegistrar()->GetNodeLowering();
}

}  // namespace compiler

namespace ir {

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device,
    lazy_tensors::Span<const Node* const> post_order,
    Util::EmissionMap emit_status) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->CreateLoweringContext(name, device, post_order, emit_status);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->CreateLoweringContext(name, device);
}

}  // namespace ir
}  // namespace torch_lazy_tensors

namespace lazy_tensors {

ComputationClient* ComputationClient::Get() {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->GetComputationClient();
}

ComputationClient* ComputationClient::GetIfInitialized() {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->GetComputationClientIfInitialized();
}

std::vector<std::string> ComputationClient::GetCompilationDevices(
    const std::string& device, lazy_tensors::Span<const std::string> devices) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->GetCompilationDevices(device, devices);
}

at::Tensor MakeTensorFromComputationData(
    const ComputationClient::DataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->MakeTensorFromComputationData(data, logical_scalar_type);
}

ComputationClient::DataPtr MakeComputationDataFromTensor(
    const at::Tensor& tensor, const Shape& shape, const std::string& device) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->MakeComputationDataFromTensor(tensor, shape, device);
}

}  // namespace lazy_tensors
