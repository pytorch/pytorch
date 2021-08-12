#pragma once

#include <atomic>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/computation_client/computation_client.h"

namespace torch_lazy_tensors {
namespace compiler {

class BackendImplInterface {
 public:
  virtual std::unique_ptr<NodeLowering> CreateNodeLowering(
      ir::LoweringContext* loctx) const = 0;

  // For inference only.
  virtual NodeLowering* GetNodeLowering() const = 0;

  virtual std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device,
      lazy_tensors::Span<const ir::Node* const> post_order,
      ir::Util::EmissionMap emit_status) const = 0;

  virtual std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device) const = 0;

  virtual lazy_tensors::ComputationClient* GetComputationClient() const = 0;

  virtual lazy_tensors::ComputationClient* GetComputationClientIfInitialized()
      const = 0;

  virtual std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      lazy_tensors::Span<const std::string> devices) const = 0;

  virtual at::Tensor MakeTensorFromComputationData(
      const lazy_tensors::ComputationClient::DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const = 0;

  virtual lazy_tensors::ComputationClient::DataPtr
  MakeComputationDataFromTensor(const at::Tensor& tensor,
                                const lazy_tensors::Shape& shape,
                                const std::string& device) const = 0;

  virtual lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const lazy_tensors::GenericComputation* computation) const = 0;

  // No-op by default. Allows custom functionality to be exposed through
  // extension bindings.
  virtual void InitializeAtenBindings() const {}
};

extern std::atomic<const BackendImplInterface*> backend_impl_registry;

class BackendRegistrar {
 public:
  BackendRegistrar(BackendImplInterface* backend_impl_interface);
};

inline const BackendImplInterface* getBackendRegistrar() {
  auto p = backend_impl_registry.load();
  LTC_CHECK(p) << "Lazy tensor backend not registered.";
  return p;
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
