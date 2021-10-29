#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

namespace torch_lazy_tensors {
namespace compiler {

std::atomic<const BackendImplInterface*> backend_impl_registry;

BackendRegistrar::BackendRegistrar(
    const BackendImplInterface* backend_impl_interface) {
  backend_impl_registry.store(backend_impl_interface);
}

std::unique_ptr<NodeLowering> NodeLowering::Create(ir::LoweringContext* loctx) {
  return getBackendRegistrar()->CreateNodeLowering(loctx);
}

NodeLowering* NodeLowering::Get() {
  return getBackendRegistrar()->GetNodeLowering();
}

std::vector<std::string> GetCompilationDevices(
    const std::string& device, c10::ArrayRef<std::string> devices) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->GetCompilationDevices(device, devices);
}

at::Tensor MakeTensorFromComputationData(
    const torch_lazy_tensors::compiler::DataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->MakeTensorFromComputationData(data, logical_scalar_type);
}

torch_lazy_tensors::compiler::DataPtr MakeComputationDataFromTensor(
    const at::Tensor& tensor, const lazy_tensors::Shape& shape,
    const std::string& device) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->MakeComputationDataFromTensor(tensor, shape, device);
}

}  // namespace compiler

namespace ir {

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device,
    c10::ArrayRef<torch::lazy::Node*> post_order,
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