#pragma once

#include <atomic>

#include <ATen/Tensor.h>
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/compiler/data.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/statusor.h"

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
      c10::ArrayRef<torch::lazy::Node*> post_order,
      ir::Util::EmissionMap emit_status) const = 0;

  virtual std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device) const = 0;

  virtual std::vector<std::string> GetCompilationDevices(
      const std::string& device, c10::ArrayRef<std::string> devices) const = 0;

  virtual at::Tensor MakeTensorFromComputationData(
      const DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const = 0;

  virtual DataPtr
  MakeComputationDataFromTensor(const at::Tensor& tensor,
                                const lazy_tensors::Shape& shape,
                                const std::string& device) const = 0;

  virtual lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const GenericComputation* computation) const = 0;

  // No-op by default. Allows custom functionality to be exposed through
  // extension bindings.
  virtual void InitializeAtenBindings() const {}

  /// computation client interfaces//////

  virtual DataPtr CreateDataPlaceholder(std::string device, lazy_tensors::Shape shape) const = 0;

  virtual std::vector<DataPtr> TransferToServer(
      c10::ArrayRef<at::Tensor> tensors) const = 0;

  virtual std::vector<at::Tensor> TransferFromServer(
      c10::ArrayRef<DataPtr> handles) const = 0;

  virtual std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) const = 0;

  virtual std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, c10::ArrayRef<DataPtr> arguments,
      const std::string& device, const ExecuteComputationOptions& options) const = 0;

  virtual std::string GetResourceDomain(const std::string& device) const = 0;

  virtual std::string GetDefaultDevice() const = 0;

  virtual size_t GetNumDevices() const = 0;

  virtual std::vector<std::string> GetLocalDevices() const = 0;

  virtual std::vector<std::string> GetAllDevices() const = 0;

  virtual void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) const = 0;

  virtual std::shared_ptr<std::vector<std::string>> GetReplicationDevices() const = 0;

  virtual void SetRngSeed(size_t seed) const = 0;

//   virtual std::map<std::string, Metric> GetMetrics() const = 0;

//   virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

  virtual void PrepareToExit() const = 0;

  virtual at::DeviceType HardwareDeviceType() const = 0;
};

extern std::atomic<const BackendImplInterface*> backend_impl_registry;

class BackendRegistrar {
 public:
  BackendRegistrar(const BackendImplInterface* backend_impl_interface);
};

// TODO(whc) do we want this to be const?
// can we implement methods like transfer to/from server if we use a const ref
inline const BackendImplInterface* getBackendRegistrar() {
  auto p = backend_impl_registry.load();
  CHECK(p) << "Lazy tensor backend not registered.";
  return p;
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
