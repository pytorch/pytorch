#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {

class TSBackendImpl : public BackendImplInterface {
 public:
  std::unique_ptr<NodeLowering> CreateNodeLowering(
      ir::LoweringContext* loctx) const override {
    return CreateTSNodeLowering(loctx);
  }

  NodeLowering* GetNodeLowering() const override { return GetTSNodeLowering(); }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device,
      lazy_tensors::Span<const ir::Node* const> post_order,
      ir::Util::EmissionMap emit_status) const override {
    return std::make_unique<ts_backend::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device) const override {
    return std::make_unique<ts_backend::TSLoweringContext>(name, device);
  }

  lazy_tensors::ComputationClient* GetComputationClient() const override {
    return lazy_tensors::compiler::TSClientGet();
  }

  lazy_tensors::ComputationClient* GetComputationClientIfInitialized()
      const override {
    return lazy_tensors::compiler::TSClientGetIfInitialized();
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      lazy_tensors::Span<const std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const lazy_tensors::ComputationClient::DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    const auto ts_data = std::static_pointer_cast<
        lazy_tensors::compiler::TSComputationClient::TSData>(data);
    return ts_data->data_;
  }

  lazy_tensors::ComputationClient::DataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const lazy_tensors::Shape& shape,
      const std::string& device) const override {
    at::TensorOptions options = tensor.options().device(
        lazy_tensors::compiler::TSComputationClient::HardwareDeviceType());
    return std::make_shared<
        lazy_tensors::compiler::TSComputationClient::TSData>(
        tensor.to(options), lazy_tensors::ToShapeData(shape), device);
  }

  lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const lazy_tensors::GenericComputation* computation) const override {
    auto ts_computation = static_cast<
        const torch_lazy_tensors::compiler::ts_backend::GenericComputationTS*>(
        computation);
    return ts_computation->graph()->toString();
  }
};

BackendImplInterface* GetTSBackendImpl() {
  static TSBackendImpl* ts_backend_impl = new TSBackendImpl();
  return ts_backend_impl;
}

void InitTorchScriptBackend() {
  static std::unique_ptr<BackendRegistrar> s_registrar;
  s_registrar.reset(new BackendRegistrar(GetTSBackendImpl()));
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
