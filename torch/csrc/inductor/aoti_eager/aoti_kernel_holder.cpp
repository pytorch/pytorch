#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/aoti_kernel_holder.h>

#include <ATen/ATen.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/inductor/aoti_eager/aoti_kernel_holder.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <regex>

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
    bool is_symbolic)
    : python_kernel_holder_(func, dispatch_key),
      dispatch_key_(dispatch_key),
      op_name_(std::string(op_name)),
      is_symbolic_(is_symbolic),
      device_opt_(c10::nullopt) {
  // TODO: To provide a registration mechanim to avoid adding such if-else block
  if (dispatch_key_ == c10::DispatchKey::CUDA) {
    device_opt_ = c10::Device(c10::DeviceType::CUDA, 0);
  } else if (dispatch_key_ == c10::DispatchKey::XPU) {
    device_opt_ = c10::Device(c10::DeviceType::XPU, 0);
  } else {
    device_opt_ = c10::Device(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
  }

  canonicalizeOpName();
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
  if (kernel_handle == aoti_kernel_cache_.end() || !kernel_handle->second) {
    python_kernel_holder_(op, keyset, stack);
    return;
  }

  // Cache hit
  torch::jit::pop(*stack, op.schema().arguments().size());
  auto aoti_eager_kernel = kernel_handle->second;
  auto outputs = (*aoti_eager_kernel)(inputs);
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
        is_symbolic_,
        input.scalar_type(),
        input.device(),
        input.sizes().vec(),
        input.strides().vec()));
  }
  return inputs_meta_info;
}

/**
 * Initializes the cache for AOTInductor kernels within the
 * AOTIPythonKernelHolder class.
 *
 * The path of AOTI kernels for eager is
 *  - ${TORCHINDUCTOR_CACHE_DIR}/${kernel_path}/${kernel_id}.so
 *
 * Besides the kernel library, there is also a metadata file for each kernel
 * library.
 *  - ${TORCHINDUCTOR_CACHE_DIR}/aten_eager/${op_name}.json
 *
 * The kernels are loaded from the path and cached in the
 * AOTIPythonKernelHolder.
 *
 * Process:
 * 1. Device Type Check: It first checks if the device type is the compile-time
 * maximum. If so, the function exits early, as no initialization is needed for
 * these device types.
 *
 * 2. Environment Variable Retrieval: Attempts to retrieve the Eager AOTI kernel
 * path from the "TORCHINDUCTOR_CACHE_DIR" environment variable. If this
 * variable isn't set, the function exits early, indicating no path is provided.
 *
 * 3. AOTI Kernel Path Construction: Constructs the path to the AOTI kernels by
 * combining the base path from the environment variable with subdirectories
 * based on the device type (cpu, cuda, xpu) and operation name. This results in
 * a specific path targeting the required AOTI kernel for the current operation
 * and device.
 *
 * 4. Path Existence Check: Checks if the constructed path exists in the file
 * system. If not, the function returns, as there are no kernels to load.
 *
 * 5. Kernel File Processing: If the path exists, iterates through each file in
 * the directory. For files with a .so extension, it replaces this with .conf to
 * locate the corresponding kernel configuration file.
 *
 * 6. Kernel Metadata Loading and Caching: Reads the kernel metadata from each
 * .conf file (using TensorMetaInfo::fromConfig). If successful, adds this
 * metadata to the aoti_kernel_cache_. The cache maps the kernel metadata to a
 * corresponding AOTI model container runner, obtained via
 * getAOTIModelContainerRunner.
 *
 * This function is crucial for setting up the AOTI kernel infrastructure,
 * enabling efficient inference operations tailored to the specific runtime
 * environment.
 */
void AOTIPythonKernelHolder::initAOTIKernelCache() {
  if (device_opt_.value().type() ==
      c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) {
    return;
  }

  fs::path eager_aoti_cache_path;
  auto inductor_cache_dir = std::getenv("TORCHINDUCTOR_CACHE_DIR");
  if (inductor_cache_dir == nullptr) {
#ifdef _WIN32
    return;
#else
    std::string username = std::getenv("USER");
    std::regex special_chars(R"([\\/:*?"<>|])");
    std::string sanitized_username =
        std::regex_replace(username, special_chars, "_");
    std::string temp_dir = std::filesystem::temp_directory_path();
    if (temp_dir.empty())
      temp_dir = std::getenv("TMPDIR");
    if (temp_dir.empty())
      temp_dir = "/tmp";
    eager_aoti_cache_path =
        fs::path(temp_dir) / fs::path("torchinductor_" + sanitized_username);
#endif
  } else {
    eager_aoti_cache_path = inductor_cache_dir;
  }

  fs::path eager_aoti_json_path =
      fs::path("aten_eager") / fs::path(op_name_ + ".json");
  fs::path eager_aoti_full_json_path =
      eager_aoti_cache_path / eager_aoti_json_path;
  if (!fs::exists(eager_aoti_full_json_path)) {
    return;
  }

  try {
    std::ifstream json_file(eager_aoti_full_json_path);
    nlohmann::json conf_json;
    json_file >> conf_json;

    for (auto& element : conf_json) {
      if (element["meta_info"] == nullptr ||
          element["kernel_path"] == nullptr) {
        continue;
      }

      std::string kernel_so_path = element["kernel_path"];
      if (!fs::exists(kernel_so_path)) {
        continue;
      }

      std::vector<std::string> tensors_meta_info = element["meta_info"];
      auto kernel_meta_info = TensorMetaInfo::fromConfig(tensors_meta_info);
      if (kernel_meta_info.size() > 0) {
        aoti_kernel_cache_[kernel_meta_info] =
            getAOTIEagerKernelRunner(kernel_so_path);
      }
    }
  } catch (nlohmann::detail::parse_error& e) {
    TORCH_CHECK(false, e.what());
  }
}

void AOTIPythonKernelHolder::canonicalizeOpName() {
  // Canonicalize the op_name as a valid directory name
  std::replace(op_name_.begin(), op_name_.end(), '.', '_');
  const std::string to_remove = "aten::";
  size_t start_pos = op_name_.find(to_remove);
  if (start_pos != std::string::npos) {
    op_name_.replace(start_pos, to_remove.length(), "");
  }
}

std::shared_ptr<AOTIEagerKernelRunner> AOTIPythonKernelHolder::
    getAOTIEagerKernelRunner(const std::string& so_path) {
  if (device_opt_.value().type() == c10::DeviceType::CUDA) {
    return std::make_shared<AOTIEagerKernelRunnerCuda>(so_path);
  } else {
    return nullptr;
  }
}

} // namespace torch::inductor
#endif