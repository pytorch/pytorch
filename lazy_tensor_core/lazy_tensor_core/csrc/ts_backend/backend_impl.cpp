#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/sys_util.h"

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
      c10::ArrayRef<torch::lazy::Node*> post_order,
      ir::Util::EmissionMap emit_status) const override {
    return std::make_unique<ts_backend::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device) const override {
    return std::make_unique<ts_backend::TSLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    const auto ts_data = std::static_pointer_cast<TSData>(data);
    return ts_data->data();
  }

  DataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const lazy_tensors::Shape& shape,
      const std::string& device) const override {
    at::TensorOptions options = tensor.options().device(HardwareDeviceType());
    return std::make_shared<TSData>(tensor.to(options), shape, device);
  }

  lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const GenericComputation* computation) const override {
    auto ts_computation = static_cast<
        const torch_lazy_tensors::compiler::ts_backend::GenericComputationTS*>(
        computation);
    return ts_computation->graph()->toString();
  }

  //////////////computation client interfaces///////////////////////

 public:
  class TSData : public Data {
   public:
    TSData(const at::Tensor& data, lazy_tensors::Shape shape,
           std::string device)
        : Data(device, shape), data_(data) {}

    TSData(lazy_tensors::Shape shape, std::string device) :Data(device, shape) {}

    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<int64_t>(this);
    }

    void Assign(const Data& data) override {
      data_ = static_cast<const TSData&>(data).data_;
    }

    bool HasValue() const override { return data_.defined(); }

    at::Tensor data() { return data_; }

   private:
    at::Tensor data_;
  };

  DataPtr CreateDataPlaceholder(std::string device,
                                lazy_tensors::Shape shape) const override;

  std::vector<DataPtr> TransferToServer(
      c10::ArrayRef<at::Tensor> tensors) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<at::Tensor> TransferFromServer(
      c10::ArrayRef<DataPtr> handles) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) const override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, c10::ArrayRef<DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) const override;

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  size_t GetNumDevices() const override { return 1; }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) const override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() const override;

  void SetRngSeed(size_t seed) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  // std::map<std::string, Metric> GetMetrics() const override { return {}; }

  // MemoryInfo GetMemoryInfo(const std::string& device) override {
  //   LOG(FATAL) << "Not implemented yet.";
  // }

  void PrepareToExit() const override;

  at::DeviceType HardwareDeviceType() const override;
};

DataPtr TSBackendImpl::CreateDataPlaceholder(std::string device,
                                             lazy_tensors::Shape shape) const {
  return std::make_shared<TSBackendImpl::TSData>(shape, std::move(device));
}

std::vector<ComputationPtr> TSBackendImpl::Compile(
    std::vector<CompileInstance> instances) const {
  std::vector<ComputationPtr> ts_computations;
  for (const auto& instance : instances) {
    auto ts_computation = static_cast<const ts_backend::GenericComputationTS*>(
        instance.computation.get());
    ProgramShape program_shape =
        ConsumeValue(ts_computation->GetProgramShape());
    ts_computations.push_back(std::make_shared<Computation>(
        instance.computation, program_shape, instance.devices));
  }
  return ts_computations;
}

std::vector<DataPtr> TSBackendImpl::ExecuteComputation(
    const Computation& computation, c10::ArrayRef<DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) const {
  torch::jit::GraphExecutor& graph_executor =
      static_cast<
          torch_lazy_tensors::compiler::ts_backend::GenericComputationTS*>(
          computation.computation())
          ->graph_executor();
  std::vector<torch::jit::IValue> stack;
  for (auto argument : arguments) {
    const auto ts_data =
        std::static_pointer_cast<TSBackendImpl::TSData>(argument);
    CHECK(HardwareDeviceType() != at::kCUDA ||
          ts_data->data().device().type() == at::kCUDA);
    stack.emplace_back(ts_data->data());
  }
  graph_executor.run(stack);
  std::vector<torch_lazy_tensors::compiler::DataPtr> results;
  for (torch::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();
    at::IntArrayRef result_sizes = result.sizes();
    lazy_tensors::Shape shape(
        result.scalar_type(),
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    results.push_back(
        std::make_shared<TSBackendImpl::TSData>(result, shape, device));
  }
  return results;
}

std::string TSBackendImpl::GetResourceDomain(const std::string& device) const {
  return "";
}

std::string TSBackendImpl::GetDefaultDevice() const {
  switch (HardwareDeviceType()) {
    case at::kCPU: {
      return "CPU:0";
    }
    case at::kCUDA: {
      return "GPU:0";
    }
    default: {
      LOG(FATAL) << "Invalid device type";
    }
  }
}

std::vector<std::string> TSBackendImpl::GetLocalDevices() const {
  return {GetDefaultDevice()};
}

std::vector<std::string> TSBackendImpl::GetAllDevices() const {
  return GetLocalDevices();
}

void TSBackendImpl::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) const {
  CHECK_EQ(devices->size(), size_t(1)) << "Replication not supported yet";
}

std::shared_ptr<std::vector<std::string>>
TSBackendImpl::GetReplicationDevices() const {
  return nullptr;
}

void TSBackendImpl::PrepareToExit() const {

}

at::DeviceType TSBackendImpl::HardwareDeviceType() const {
  static auto device_type =
      lazy_tensors::sys_util::GetEnvBool("LTC_TS_CUDA", false) ? at::kCUDA : at::kCPU;
  // The first CUDA usage could happen via lazy tensors. Initialize CUDA here to
  // account for that, at::scalar_tensor constructor triggers everything we
  // need.
  static c10::optional<at::Tensor> init_cuda =
      device_type == at::kCUDA ? c10::optional<at::Tensor>(at::scalar_tensor(
                                     0, at::TensorOptions().device(at::kCUDA)))
                               : c10::nullopt;
  return device_type;
}

compiler::BackendImplInterface* GetTSBackendImpl() {
  static compiler::TSBackendImpl* ts_backend_impl =
      new compiler::TSBackendImpl();
  return ts_backend_impl;
}

void InitTorchScriptBackend() {
  static std::unique_ptr<compiler::BackendRegistrar> s_registrar;
  s_registrar.reset(new compiler::BackendRegistrar(compiler::GetTSBackendImpl()));
}
};  // namespace compiler
}  // namespace torch_lazy_tensors
