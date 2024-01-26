#include <torch/csrc/inductor/aoti_eager/aoti_kernel_holder.h>

#include <ATen/ATen.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/inductor/aoti_eager/aoti_kernel_holder.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace torch::inductor {

namespace {

enum class IValueType : uint8_t {
  Tensor,
  TensorList,
  OptionalTensorList,
  Scalar,
  Invalid,
};

static bool HandleTensor(
    const c10::IValue& ivalue,
    std::vector<at::Tensor>& inputs,
    c10::Device& device) {
  inputs.push_back(ivalue.toTensor());
  return true;
}

static bool HandleTensorList(
    const c10::IValue& ivalue,
    std::vector<at::Tensor>& inputs,
    c10::Device& device) {
  for (const auto& item : ivalue.toListRef()) {
    if (!item.isNone()) {
      inputs.push_back(item.toTensor());
    }
  }
  return true;
}

static bool HandleOptionalTensorList(
    const c10::IValue& ivalue,
    std::vector<at::Tensor>& inputs,
    c10::Device& device) {
  return HandleTensorList(ivalue, inputs, device);
}

static bool HandleScalar(
    const c10::IValue& ivalue,
    std::vector<at::Tensor>& inputs,
    c10::Device& device) {
  inputs.push_back(at::scalar_tensor(
      ivalue.toScalar(),
      c10::TensorOptions().device(device).dtype(ivalue.toScalar().type())));
  return true;
}

typedef bool (*HandlerFunc)(
    const c10::IValue& ivalue,
    std::vector<at::Tensor>& inputs,
    c10::Device& device);
std::unordered_map<IValueType, HandlerFunc> handlers_ = {
    {IValueType::Tensor, &HandleTensor},
    {IValueType::TensorList, &HandleTensorList},
    {IValueType::OptionalTensorList, &HandleOptionalTensorList},
    {IValueType::Scalar, &HandleScalar}};

bool HandleIValue(
    const c10::IValue& ivalue,
    std::vector<at::Tensor>& inputs,
    c10::Device& device) {
  IValueType ivalue_type = IValueType::Invalid;
  if (ivalue.isTensor()) {
    ivalue_type = IValueType::Tensor;
  } else if (ivalue.isTensorList()) {
    ivalue_type = IValueType::TensorList;
  } else if (ivalue.isOptionalTensorList()) {
    ivalue_type = IValueType::OptionalTensorList;
  } else if (ivalue.isScalar()) {
    ivalue_type = IValueType::Scalar;
  }

  auto it = handlers_.find(ivalue_type);
  if (it != handlers_.end()) {
    return it->second(ivalue, inputs, device);
  }

  // Handle unsupported types or add a default handler
  return false;
}

} // namespace

AOTIPythonKernelHolder::AOTIPythonKernelHolder(
    py::object func,
    c10::DispatchKey dispatch_key,
    c10::string_view op_name,
    bool is_dynamic)
    : python_kernel_holder_(func, dispatch_key),
      dispatch_key_(dispatch_key),
      op_name_(op_name),
      is_dynamic_(is_dynamic),
      device_opt_(c10::nullopt) {
  if (dispatch_key_ == c10::DispatchKey::CUDA) {
    device_opt_ = c10::Device(c10::DeviceType::CUDA, 0);
  } else if (dispatch_key_ == c10::DispatchKey::XPU) {
    device_opt_ = c10::Device(c10::DeviceType::XPU, 0);
  } else {
    device_opt_ = c10::Device(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
  }

  initAOTIKernelCache();
}

void AOTIPythonKernelHolder::operator()(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  std::vector<at::Tensor> inputs;
  auto res = unpackTensors(*stack, inputs);
  if (!res || inputs.empty()) {
    python_kernel_holder_(op, keyset, stack);
    return;
  }

  auto inputs_meta_info = getInputsMetaInfo(inputs);
  auto kernel_handle = aoti_kernel_cache_.find(inputs_meta_info);
  // Cache miss
  if (kernel_handle == aoti_kernel_cache_.end()) {
    python_kernel_holder_(op, keyset, stack);
    return;
  }

  // Cache hit
  torch::jit::pop(*stack, op.schema().arguments().size());
  auto outputs = kernel_handle->second->run(inputs);
  for (auto& output : outputs) {
    stack->push_back(output);
  }
}

bool AOTIPythonKernelHolder::unpackTensors(
    const torch::jit::Stack& stack,
    std::vector<at::Tensor>& inputs) {
  for (const auto& ivalue : stack) {
    if (!HandleIValue(ivalue, inputs, device_opt_.value())) {
      return false;
    }
  }
  return true;
}

AOTIKernelMetaInfo AOTIPythonKernelHolder::getInputsMetaInfo(
    const std::vector<at::Tensor>& inputs) {
  AOTIKernelMetaInfo inputs_meta_info;
  for (const auto& input : inputs) {
    inputs_meta_info.push_back(TensorMetaInfo(
        input.scalar_type(),
        input.device(),
        input.sizes().vec(),
        input.strides().vec()));
  }
  return inputs_meta_info;
}

void AOTIPythonKernelHolder::initAOTIKernelCache() {
  if (device_opt_.value().type() ==
      c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) {
    return;
  }

  auto eager_aoti_kernel_path = std::getenv("TORCH_EAGER_AOTI_KERNEL_PATH");
  if (eager_aoti_kernel_path == nullptr) {
    return;
  }

  fs::path eager_aoti_path = eager_aoti_kernel_path;
  fs::path eager_aoti_device_path =
      device_opt_.value().type() == c10::DeviceType::CPU    ? "cpu"
      : device_opt_.value().type() == c10::DeviceType::CUDA ? "cuda"
                                                            : "xpu";
  fs::path eager_aoti_op_path = std::string(op_name_);
  fs::path eager_aoti_full_path =
      eager_aoti_path / eager_aoti_device_path / eager_aoti_op_path;
  if (!fs::exists(eager_aoti_full_path)) {
    return;
  }

  for (auto const& entry : fs::directory_iterator(eager_aoti_full_path)) {
    fs::path so_path = entry.path().string();
    if (so_path.extension() == ".so") {
      fs::path& kernel_conf = so_path.replace_extension(".conf");
      auto kernel_meta_infos = TensorMetaInfo::fromConfig(kernel_conf);
      if (kernel_meta_infos.size() > 0) {
        aoti_kernel_cache_[kernel_meta_infos] =
            getAOTIModelContainerRunner(so_path);
      }
    }
  }
}

std::shared_ptr<AOTIModelContainerRunner> AOTIPythonKernelHolder::
    getAOTIModelContainerRunner(const std::string& so_path) {
  if (device_opt_.value().type() == c10::DeviceType::CUDA) {
    return std::make_shared<AOTIModelContainerRunnerCuda>(so_path);
  } else {
    return nullptr;
  }
}

} // namespace torch::inductor
