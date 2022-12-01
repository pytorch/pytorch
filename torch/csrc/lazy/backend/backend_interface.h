#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <atomic>

namespace torch {
namespace lazy {

struct IrBuilder;

/**
 * Work in progress- don't treat this as a stable interface yet!
 */
class TORCH_API BackendImplInterface {
 public:
  virtual ~BackendImplInterface() = default;

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
   * IR Tracing
   * */

  virtual const IrBuilder* GetIrBuilder() const = 0;

  /**
   * Data Transfer
   * */

  virtual BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor,
      const Shape& shape,
      const BackendDevice& device) const = 0;
  virtual BackendDataPtr MakeComputationDataFromScalar(
      const at::Scalar& scalar,
      const torch::lazy::BackendDevice& device) const = 0;
  virtual BackendDataPtr CreateDataPlaceholder(
      const BackendDevice& device,
      const Shape& shape) const = 0;

  // Gets backend data if the node is a device data node. Otherwise returns
  // nullptr
  virtual BackendDataPtr GetComputationDataFromNode(const Node*) const = 0;

  virtual at::Tensor MakeTensorFromComputationData(
      const BackendDataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const = 0;

  /**
   * Lowering, Compilation, Execution
   * */

  virtual std::unique_ptr<LoweringContext> CreateLoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status) const = 0;

  virtual std::unique_ptr<LoweringContext> CreateLoweringContext(
      const std::string& name,
      BackendDevice device) const = 0;

  // TODO(whc) need to keep this?
  virtual std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const = 0;

  virtual std::vector<ComputationPtr> Compile(
      std::vector<ComputationPtr> instances) const = 0;

  virtual std::vector<BackendDataPtr> ExecuteComputation(
      torch::lazy::ComputationPtr computation,
      c10::ArrayRef<BackendDataPtr> arguments,
      const BackendDevice& device) const = 0;

  /**
   * Device Configuration
   * */

  // Set or get the default device type.
  // For backends used with virtual c10::Devices, this configures what real
  // device type the backend should use, and matters if the backend supports
  // more than one type of real device.
  virtual std::shared_ptr<BackendDeviceType> GetDefaultDeviceType() const = 0;
  virtual void SetDefaultDeviceType(int8_t type) = 0;

  // Set or get the default device ordinal.
  // For backends that supports multi-device, this configures what the
  // default device the backend should use.
  virtual int64_t GetDefaultDeviceOrdinal() const = 0;
  virtual void SetDefaultDeviceOrdinal(int64_t) = 0;

  // Specify which aten device should be used for eager fallback
  // may change depending on current 'Default' DeviceType
  virtual at::DeviceType EagerFallbackDeviceType() const = 0;

  // Query all available backend devices
  virtual std::vector<BackendDevice> GetBackendDevices() const = 0;

  virtual std::string CreateMetricReport() const {
    return "";
  }

  // Map a particular c10:: device to a concrete backend device
  // Note:: c10:: devices may be virtual or concrete.  xla:: and lazy:: are
  // virtual devices, meaning they may map to a gpu, tpu, etc. behind the
  // scenes. In the future, non-virtual c10:: devices may also use lazy tensors
  // through a mode, in which case these APIs should still work, but should be
  // identity mappings.
  virtual BackendDevice GetBackendDevice(c10::Device device) const = 0;

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

class TORCH_API BackendRegistrar {
 public:
  BackendRegistrar(const BackendImplInterface* backend_impl_interface);
};

TORCH_API bool hasBackend();
TORCH_API const BackendImplInterface* getBackend();

TORCH_API const IrBuilder* getIrBuilder();

} // namespace lazy
} // namespace torch
