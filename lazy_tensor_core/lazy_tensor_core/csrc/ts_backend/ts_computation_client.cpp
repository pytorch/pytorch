#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

namespace lazy_tensors {
namespace {

std::atomic<lazy_tensors::ComputationClient*> g_computation_client(nullptr);
std::once_flag g_computation_client_once;

lazy_tensors::ComputationClient* CreateClient() {
  return new compiler::TSComputationClient();
}

}  // namespace

namespace compiler {

ComputationClient::DataPtr TSComputationClient::CreateDataPlaceholder(
    std::string device, Shape shape) {
  return std::make_shared<TSComputationClient::TSData>(
      lazy_tensors::ToShapeData(shape), std::move(device));
}

std::vector<ComputationClient::ComputationPtr> TSComputationClient::Compile(
    std::vector<CompileInstance> instances) {
  std::vector<ComputationClient::ComputationPtr> ts_computations;
  for (const auto& instance : instances) {
    auto ts_computation = static_cast<
        const torch_lazy_tensors::compiler::ts_backend::GenericComputationTS*>(
        instance.computation.get());
    ProgramShape program_shape =
        ConsumeValue(ts_computation->GetProgramShape());
    ts_computations.push_back(std::make_shared<Computation>(
        instance.computation, program_shape, instance.devices));
  }
  return ts_computations;
}

std::vector<ComputationClient::DataPtr> TSComputationClient::ExecuteComputation(
    const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  auto graph =
      static_cast<
          torch_lazy_tensors::compiler::ts_backend::GenericComputationTS*>(
          computation.computation())
          ->graph();
  torch::jit::GraphExecutor interp(graph, "");
  std::vector<torch::jit::IValue> stack;
  for (auto argument : arguments) {
    const auto ts_data =
        std::static_pointer_cast<TSComputationClient::TSData>(argument);
    LTC_CHECK(
        lazy_tensors::compiler::TSComputationClient::HardwareDeviceType() !=
            at::kCUDA ||
        ts_data->data_.device().type() == at::kCUDA);
    stack.emplace_back(ts_data->data_);
  }
  interp.run(stack);
  std::vector<ComputationClient::DataPtr> results;
  for (torch::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();
    at::IntArrayRef result_sizes = result.sizes();
    lazy_tensors::PrimitiveType element_type =
        torch_lazy_tensors::TensorTypeToLtcType(result.scalar_type());
    client::ShapeData shape(
        element_type,
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    results.push_back(
        std::make_shared<TSComputationClient::TSData>(result, shape, device));
  }
  return results;
}

std::string TSComputationClient::GetResourceDomain(
    const std::string& device) const {
  return "";
}

std::string TSComputationClient::GetDefaultDevice() const {
  switch (lazy_tensors::compiler::TSComputationClient::HardwareDeviceType()) {
    case at::kCPU: {
      return "CPU:0";
    }
    case at::kCUDA: {
      return "GPU:0";
    }
    default: { LTC_LOG(FATAL) << "Invalid device type"; }
  }
}

std::vector<std::string> TSComputationClient::GetLocalDevices() const {
  return {GetDefaultDevice()};
}

std::vector<std::string> TSComputationClient::GetAllDevices() const {
  return GetLocalDevices();
}

void TSComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  LTC_CHECK_EQ(devices->size(), size_t(1)) << "Replication not supported yet";
}

std::shared_ptr<std::vector<std::string>>
TSComputationClient::GetReplicationDevices() {
  return nullptr;
}

void TSComputationClient::PrepareToExit() {}

at::DeviceType TSComputationClient::HardwareDeviceType() {
  static auto device_type =
      sys_util::GetEnvBool("LTC_TS_CUDA", false) ? at::kCUDA : at::kCPU;
  // The first CUDA usage could happen via lazy tensors. Initialize CUDA here to
  // account for that, at::scalar_tensor constructor triggers everything we
  // need.
  static c10::optional<at::Tensor> init_cuda =
      device_type == at::kCUDA ? c10::optional<at::Tensor>(at::scalar_tensor(
                                     0, at::TensorOptions().device(at::kCUDA)))
                               : c10::nullopt;
  return device_type;
}

lazy_tensors::ComputationClient* TSClientGet() {
  std::call_once(g_computation_client_once,
                 [&]() { g_computation_client = CreateClient(); });
  return g_computation_client.load();
}

lazy_tensors::ComputationClient* TSClientGetIfInitialized() {
  return g_computation_client.load();
}

}  // namespace compiler
}  // namespace lazy_tensors
