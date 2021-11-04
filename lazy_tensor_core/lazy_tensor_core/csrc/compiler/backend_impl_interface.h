#pragma once

#include <atomic>
#include <ATen/Tensor.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/backend/backend_device.h>

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

  // Set or get the default device type.
  // For backends used with virtual c10:: Devices, this configures what real
  // device type the backend should use, and matters if the backend supports
  // more than one type of real device.
  virtual std::shared_ptr<torch::lazy::BackendDeviceType>
  GetDefaultDeviceType() const = 0;
  virtual void SetDefaultDeviceType(std::string) = 0;

  // Specify which aten device should be used for eager fallback
  // may change depending on current 'Default' DeviceType
  virtual at::DeviceType EagerFallbackDeviceType() const = 0;


  // Query all available backend devices
  virtual std::vector<torch::lazy::BackendDevice> GetBackendDevices() const = 0;

  // Map a particular c10:: device to a concrete backend device
  // Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
  // virtual devices, meaning they may map to a gpu, tpu, etc. behind the
  // scenes. In the future, non-virtual c10:: devices may also use lazy tensors
  // through a mode, in which case these APIs should still work, but should be
  // identity mappings.
  virtual torch::lazy::BackendDevice GetBackendDevice(
      c10::Device device) const = 0;

  // TODO(whc)
  // Additional APIs expected for supporting distributed training, to be
  // designed

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
