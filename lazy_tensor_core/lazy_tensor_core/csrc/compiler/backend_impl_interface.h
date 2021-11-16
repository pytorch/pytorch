#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/shape.h>

#include <atomic>

#include "lazy_tensor_core/csrc/lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

class BackendImplInterface {
 public:
  /**
   * Initialization/Teardown
   * */
  // No-op by default. Allows custom functionality to be exposed through
  // extension bindings.
  virtual void InitializeAtenBindings() const {}

  virtual void PrepareToExit() const = 0;

  /**
   * Configuration
   * */

  virtual void SetRngSeed(size_t seed) const = 0;

  /**
   * Data Transfer
   * */

  virtual torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device) const = 0;

  virtual torch::lazy::BackendDataPtr CreateDataPlaceholder(
      const torch::lazy::BackendDevice& device,
      const torch::lazy::Shape& shape) const = 0;

  virtual at::Tensor MakeTensorFromComputationData(
      const torch::lazy::BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const = 0;

  /**
   * Lowering, Compilation, Execution
   * */

  virtual std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, torch::lazy::BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const = 0;

  virtual std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, torch::lazy::BackendDevice device) const = 0;

  // TODO(whc) need to keep this?
  virtual std::vector<std::string> GetCompilationDevices(
      const std::string& device, c10::ArrayRef<std::string> devices) const = 0;

  virtual std::vector<ComputationPtr> Compile(
      std::vector<ComputationPtr> instances) const = 0;

  virtual std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
      Computation& computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device) const = 0;

  /**
   * Device Configuration
   * */

  virtual std::string GetDefaultDevice() const = 0;

  virtual size_t GetNumDevices() const = 0;

  // TODO: Return std::vector<torch::lazy::BackendDevice> instead.
  virtual std::vector<std::string> GetLocalDevices() const = 0;

  virtual std::vector<std::string> GetAllDevices() const = 0;

  virtual void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) const = 0;

  virtual std::shared_ptr<std::vector<std::string>> GetReplicationDevices()
      const = 0;

  virtual at::DeviceType HardwareDeviceType() const = 0;

  /**
   * Debug/Metrics
   * */

  //   virtual std::map<std::string, Metric> GetMetrics() const = 0;

  //   virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

  virtual std::string GetComputationBackendText(
      const ComputationPtr computation) const = 0;
};

extern std::atomic<const BackendImplInterface*> backend_impl_registry;

class BackendRegistrar {
 public:
  BackendRegistrar(const BackendImplInterface* backend_impl_interface);
};

// TODO(whc) do we want this to be const?
// can we implement methods like transfer to/from server if we use a const ref
inline const BackendImplInterface* getBackend() {
  auto p = backend_impl_registry.load();
  CHECK(p) << "Lazy tensor backend not registered.";
  return p;
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
