#include "torch/csrc/nativert/executor/AOTInductorModelImpl.h" // @manual
// TODO Always use OSS proxy executor.
#ifdef FBCODE_CAFFE2
#include "deeplearning/aot_inductor/fb/FbProxyExecutor.h"
#else
#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h> // @manual
#endif
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h> // @manual

#include <type_traits>

#include <dlfcn.h>
#include <cstdlib> // for getenv
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "ATen/Context.h" // @manual
#if defined(__SIGRID_USE_GPU__)
#include "ATen/cuda/CUDAContext.h" // @manual
#include "c10/cuda/CUDAStream.h" // @manual
#endif // __SIGRID_USE_GPU__

#include "torch/csrc/nativert/common/FileUtil.h"

namespace torch::aot_inductor {

namespace {
template <typename T>
struct GetLastArgType;

template <typename T>
struct tag {
  using type = T;
};

template <typename Function, typename... Args>
struct GetLastArgType<Function(Args...)> {
  using last_arg_type = typename decltype((tag<Args>{}, ...))::type;
};

template <typename T>
struct AOTInductorCallImpl;

template <typename... Args>
struct AOTInductorCallImpl<
    AOTIRuntimeError(AOTInductorModelContainerHandle*, Args...)> {
  // Special version for ModelContainer creation
  void operator()(
      AOTIRuntimeError (*f)(AOTInductorModelContainerHandle*, Args...),
      AOTInductorModelContainerHandle* handle,
      Args... args) {
    AOTI_RUNTIME_ERROR_CODE_CHECK(f(handle, args...));
  }
};

template <typename... Args>
struct AOTInductorCallImpl<
    AOTIRuntimeError(AOTInductorModelContainerHandle, Args...)> {
  using Function = AOTIRuntimeError(AOTInductorModelContainerHandle, Args...);
  template <typename... ArgsWithoutLastArgument>
  auto operator()(
      Function* f,
      AOTInductorModelContainerHandle handle,
      ArgsWithoutLastArgument... args) {
    std::remove_pointer_t<typename GetLastArgType<Function>::last_arg_type>
        result;
    AOTI_RUNTIME_ERROR_CODE_CHECK(f(handle, args..., &result));
    return result;
  }
  void operator()(
      AOTIRuntimeError (*f)(AOTInductorModelContainerHandle, Args...),
      AOTInductorModelContainerHandle handle,
      Args... args) {
    AOTI_RUNTIME_ERROR_CODE_CHECK(f(handle, args...));
  }
};

template <typename Function, typename... Args>
auto AOTInductorCall(
    Function* f,
    AOTInductorModelContainerHandle handle,
    Args... args) {
  return AOTInductorCallImpl<Function>()(f, handle, args...);
}

template <typename Function>
auto AOTInductorCallCreate(
    Function* f,
    AOTInductorModelContainerHandle* handle,
    size_t num_runtimes,
    bool is_cpu,
    const char* cubin_dir) {
  return AOTInductorCallImpl<Function>()(
      f, handle, num_runtimes, is_cpu, cubin_dir);
}

template <typename Function>
auto AOTInductorCallCreateWithDevice(
    Function* f,
    AOTInductorModelContainerHandle* handle,
    size_t num_runtimes,
    const char* device_str,
    const char* cubin_dir) {
  return AOTInductorCallImpl<Function>()(
      f, handle, num_runtimes, device_str, cubin_dir);
}
std::string getFileBasename(const std::string& filename) {
  const auto slash = filename.rfind('/');
  return slash != std::string::npos ? filename.substr(slash + 1) : filename;
}

// TODO: can we simply use std::filesystem::exists?
inline bool fileExists(const std::string& name) {
  const auto fd =
      torch::nativert::openNoInt(name.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd == -1) {
    return false;
  }
  torch::nativert::closeNoInt(fd);
  return true;
}

std::unique_ptr<ProxyExecutor> makeProxyExecutor(
    const std::string& filename,
    bool is_cpu,
    std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs) {
#ifdef FBCODE_CAFFE2
  return std::make_unique<FbProxyExecutor>(
      filename, is_cpu, std::move(custom_objs));
#else
  return std::make_unique<OSSProxyExecutor>(filename, is_cpu);
#endif
}
} // namespace

// private static
std::vector<std::string> AOTInductorModelImpl::library_search_paths_;

AOTInductorModelImpl::AOTInductorModelImpl(
    const std::string& model_path,
    std::optional<std::string> cubin_dir,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    std::optional<at::ScalarType> input_dtype,
    std::optional<at::ScalarType> output_dtype,
    std::optional<std::string> extern_kernel_nodes_path,
    const std::string& device_str,
    int64_t num_runtimes,
    std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs)
    : // handle_(dlopen(model_path.c_str(), RTLD_LAZY | RTLD_LOCAL)),
      libraryBasename_(getFileBasename(model_path)),
      libraryPath_(model_path),
      inputNames_(std::move(input_names)),
      outputNames_(std::move(output_names)),
      floatingPointInputDtype_(input_dtype),
      floatingPointOutputDtype_(output_dtype),
      deviceStr_(device_str) {
  LOG(INFO) << "Loading .so lib from " << model_path
            << " onto device: " << device_str;
  handle_.reset(dlopen(model_path.c_str(), RTLD_NOW | RTLD_LOCAL));
  TORCH_CHECK(
      handle_ != nullptr, "could not dlopen ", model_path, ": ", dlerror());
  TORCH_CHECK(num_runtimes > 0, "num_runtimes must be positive");

  if (extern_kernel_nodes_path.has_value()) {
    const std::string& filename = extern_kernel_nodes_path.value();
    if (fileExists(filename)) {
      LOG(INFO) << "Loading extern_kernel_nodes .json file from " << filename;

      proxyExecutor_ =
          makeProxyExecutor(filename, is_cpu(), std::move(custom_objs));
    }
  }

#if defined(__SIGRID_USE_GPU__)
  // It's not clear what stream we want to use yet. Create a new one.
  // We could alternatively use the default stream, but that could cause extra
  // synchronization.
  using StreamGuard = std::unique_ptr<
      std::remove_pointer_t<cudaStream_t>,
      decltype(&cudaStreamDestroy)>;

  std::optional<StreamGuard> creation_stream_guard = [&] {
    if (is_cpu()) {
      return std::optional<StreamGuard>();
    }
    cudaStream_t creation_stream;
    TORCH_CHECK(
        cudaStreamCreateWithFlags(&creation_stream, cudaStreamNonBlocking) ==
        cudaSuccess);
    return std::make_optional<StreamGuard>(creation_stream, cudaStreamDestroy);
  }();
#endif // __SIGRID_USE_GPU__

#define LOAD_SYMBOL(var, name_str)                                       \
  var = reinterpret_cast<decltype(var)>(dlsym(handle_.get(), name_str)); \
  TORCH_CHECK(var, "could not dlsym " name_str);

  LOAD_SYMBOL(deleteFunc_, "AOTInductorModelContainerDelete");
  LOAD_SYMBOL(runFunc_, "AOTInductorModelContainerRun");
  LOAD_SYMBOL(getOutputNameFunc_, "AOTInductorModelContainerGetOutputName");
  LOAD_SYMBOL(getCallSpecFunc_, "AOTInductorModelContainerGetCallSpec");

  // We never call these functions again after the constructor returns, so
  // there's no point in caching them in member variables.
  decltype(&AOTInductorModelContainerCreate) createFunc;
  decltype(&AOTInductorModelContainerGetInputName) getInputNameFunc;
  decltype(&AOTInductorModelContainerGetNumInputs) getNumInputsFunc;
  decltype(&AOTInductorModelContainerGetNumOutputs) getNumOutputsFunc;
  LOAD_SYMBOL(createFunc, "AOTInductorModelContainerCreate");
  LOAD_SYMBOL(getInputNameFunc, "AOTInductorModelContainerGetInputName");
  LOAD_SYMBOL(getNumInputsFunc, "AOTInductorModelContainerGetNumInputs");
  LOAD_SYMBOL(getNumOutputsFunc, "AOTInductorModelContainerGetNumOutputs");
#undef LOAD_SYMBOL

#define LOAD_SYMBOL_WARN(var, name_str)                                  \
  var = reinterpret_cast<decltype(var)>(dlsym(handle_.get(), name_str)); \
  if (!var) {                                                            \
    LOG(WARNING) << "Could not dlsym " << name_str;                      \
  }

  // "AOTInductorModelContainerCreateWithDevice" is only available in the binary
  // compiled after Jan.15.2024
  decltype(&AOTInductorModelContainerCreateWithDevice) createFuncWithDevice;
  LOAD_SYMBOL_WARN(
      createFuncWithDevice, "AOTInductorModelContainerCreateWithDevice");

  LOAD_SYMBOL_WARN(
      getNumConstantsFunc_, "AOTInductorModelContainerGetNumConstants");
  LOAD_SYMBOL_WARN(
      getConstantNameFunc_, "AOTInductorModelContainerGetConstantName");
  LOAD_SYMBOL_WARN(
      getConstantOriginalFQNFunc_,
      "AOTInductorModelContainerGetConstantOriginalFQN");
  LOAD_SYMBOL_WARN(
      getConstantFromFoldedFunc_,
      "AOTInductorModelContainerGetConstantFromFolded");
  LOAD_SYMBOL_WARN(
      getConstantTypeFunc_, "AOTInductorModelContainerGetConstantType");
  LOAD_SYMBOL_WARN(
      getConstantDtypeFunc_, "AOTInductorModelContainerGetConstantDtype");
  LOAD_SYMBOL_WARN(
      runConstantFoldingFunc_, "AOTInductorModelContainerRunConstantFolding");
  LOAD_SYMBOL_WARN(
      updateConstantBufferFunc_,
      "AOTInductorModelContainerUpdateConstantBuffer");
  LOAD_SYMBOL_WARN(
      updateInactiveConstantBufferFunc_,
      "AOTInductorModelContainerUpdateInactiveConstantBuffer");
  LOAD_SYMBOL_WARN(
      swapConstantBufferFunc_, "AOTInductorModelContainerSwapConstantBuffer");
#undef LOAD_SYMBOL_WARN

  if (createFuncWithDevice) {
    AOTInductorCallCreateWithDevice(
        createFuncWithDevice,
        &containerHandle_,
        num_runtimes,
        deviceStr_.c_str(),
        cubin_dir ? cubin_dir->c_str() : nullptr);
  } else {
    AOTInductorCallCreate(
        createFunc,
        &containerHandle_,
        num_runtimes,
        is_cpu(),
        cubin_dir ? cubin_dir->c_str() : nullptr);
  }

  const auto num_inputs = AOTInductorCall(getNumInputsFunc, containerHandle_);
  const auto num_outputs = AOTInductorCall(getNumOutputsFunc, containerHandle_);
  TORCH_CHECK(
      inputNames_.size() == num_inputs,
      "the size of input_names is ",
      inputNames_.size(),
      ", but the model expects ",
      num_inputs);
  TORCH_CHECK(
      outputNames_.size() == num_outputs,
      "the size of output_names is ",
      outputNames_.size(),
      ", but the model expects ",
      num_outputs);

  for (const auto idx : c10::irange(num_inputs)) {
    inputNameToIndex_.emplace(
        AOTInductorCall(getInputNameFunc, containerHandle_, idx), idx);
  }
  for (const auto idx : c10::irange(num_outputs)) {
    outputNameToIndex_.emplace(
        AOTInductorCall(getOutputNameFunc_, containerHandle_, idx), idx);
  }
}

std::vector<torch::Tensor> AOTInductorModelImpl::processInputs(
    std::vector<torch::Tensor>& python_inputs) {
  RECORD_USER_SCOPE("AOTInductorModel::ProcessInputs");
  const auto num_inputs = inputNameToIndex_.size();
  TORCH_CHECK(
      python_inputs.size() == num_inputs,
      "User passed ",
      python_inputs.size(),
      " inputs, but the model expects ",
      num_inputs);
  std::vector<torch::Tensor> inputs(python_inputs.size());
  for (int python_input_idx = 0; python_input_idx < inputNames_.size();
       python_input_idx++) {
    auto input_name = inputNames_[python_input_idx];
    auto& input = python_inputs[python_input_idx];
    if (floatingPointInputDtype_ != std::nullopt && input.is_floating_point()) {
      // Need to keep input alive; cannot just stash result of to()
      // call in a local!
      input = input.to(*floatingPointInputDtype_);
    }
    // FIXME: get currect aot_input_idx once we figure out name-mapping
    // in AOTInductor.
    // Currently, we have strong assumption that python_input_idx
    // (fx inputs) is the same as aot_input_idx.
    // const auto aot_input_idx = input_name_to_index_.at(input_name);
    const auto aot_input_idx = python_input_idx;
    // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
    inputs[aot_input_idx] = input;
  }
  return inputs;
}

std::vector<torch::Tensor> AOTInductorModelImpl::processOutputs(
    std::vector<torch::Tensor>&& outputs) {
  if (floatingPointOutputDtype_.has_value()) {
    for (auto& output : outputs) {
      if (output.is_floating_point()) {
        output = output.to(*floatingPointOutputDtype_);
      }
    }
  }
  return std::move(outputs);
}

std::vector<torch::Tensor> AOTInductorModelImpl::forward(
    std::vector<torch::Tensor>& python_inputs) {
  RECORD_USER_SCOPE("AOTInductorModel::Forward");
  TORCH_CHECK(!python_inputs.empty());

  std::vector<torch::Tensor> input_tensors = processInputs(python_inputs);

  // For outputs, we only allocate a vector to hold returned tensor handles,
  // not allocating the actual output tensor storage here
  const auto num_outputs = outputNameToIndex_.size();
  std::vector<AtenTensorHandle> output_handles(num_outputs);

  {
    auto input_handles =
        torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(
            input_tensors);
#if defined(__SIGRID_USE_GPU__)
    const auto device = python_inputs[0].device();
    AOTInductorStreamHandle stream_handle = is_cpu() ? nullptr : [&] {
      const auto& cuda_stream = at::cuda::getCurrentCUDAStream(device.index());
      const auto stream_id = cuda_stream.stream();
      return reinterpret_cast<AOTInductorStreamHandle>(stream_id);
    }();
#else
    AOTInductorStreamHandle stream_handle = nullptr;
#endif // __SIGRID_USE_GPU__
    AOTIProxyExecutorHandle proxy_executor_handle =
        reinterpret_cast<AOTIProxyExecutorHandle>(proxyExecutor_.get());

    RECORD_USER_SCOPE("AOTInductorModel::AOTInductorRuntime");
    AOTIRuntimeError run_result = runFunc_(
        containerHandle_,
        input_handles.data(),
        input_tensors.size(),
        output_handles.data(),
        output_handles.size(),
        stream_handle,
        proxy_executor_handle);
    if (run_result != AOTI_RUNTIME_SUCCESS) {
      std::stringstream ss;
      ss << "AOTInductorModel run failed with input spec: ";
      for (const auto& i : python_inputs) {
        ss << i.sizes() << ":" << i.dtype() << ", ";
      }
      TORCH_CHECK(false, ss.str());
    }

    return processOutputs(
        torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            output_handles.data(), output_handles.size()));
  }
}

std::vector<const char*> AOTInductorModelImpl::get_call_spec() {
  std::vector<const char*> call_spec = {nullptr, nullptr};
  getCallSpecFunc_(containerHandle_, call_spec.data(), &call_spec[1]);
  return call_spec;
}

std::unordered_map<std::string, ConstantInfo>
AOTInductorModelImpl::getConstantInfos() const {
  TORCH_CHECK(
      getNumConstantsFunc_, "getNumConstantsFunc_ was not loaded from .so");
  TORCH_CHECK(
      getConstantNameFunc_, "getConstantNameFunc_ was not loaded from .so");
  TORCH_CHECK(
      getConstantOriginalFQNFunc_,
      "getConstantOriginalFQNFunc_ was not loaded from .so");
  TORCH_CHECK(
      getConstantDtypeFunc_, "getConstantDtypeFunc_ was not loaded from .so");

  std::unordered_map<std::string, ConstantInfo> result;
  auto num_constants = AOTInductorCall(getNumConstantsFunc_, containerHandle_);
  for (size_t i = 0; i < num_constants; ++i) {
    const auto name =
        AOTInductorCall(getConstantNameFunc_, containerHandle_, i);
    const auto original_fqn =
        AOTInductorCall(getConstantOriginalFQNFunc_, containerHandle_, i);
    const auto dtype =
        AOTInductorCall(getConstantDtypeFunc_, containerHandle_, i);

    ConstantType constant_type = ConstantType::Unknown;
    if (getConstantTypeFunc_) {
      constant_type = static_cast<ConstantType>(
          AOTInductorCall(getConstantTypeFunc_, containerHandle_, i));
    }
    if (getConstantFromFoldedFunc_ &&
        AOTInductorCall(getConstantFromFoldedFunc_, containerHandle_, i)) {
      continue;
    }
    TORCH_CHECK(original_fqn, "Cannot find orignal FQN of constant ", name);

    result.emplace(
        name,
        ConstantInfo{
            static_cast<at::ScalarType>(dtype), original_fqn, constant_type});
  }
  return result;
}

void AOTInductorModelImpl::runConstantFolding(bool use_inactive) {
  if (!runConstantFoldingFunc_) {
    // We will just return if runtime constant folding doesn't exist.
    // Only models compiled after 2024 Feb has such capability.
    return;
  }

#if defined(__SIGRID_USE_GPU__)
  AOTInductorStreamHandle stream_handle = is_cpu() ? nullptr : [&] {
    const auto& cuda_stream = at::cuda::getCurrentCUDAStream();
    const auto stream_id = cuda_stream.stream();
    return reinterpret_cast<AOTInductorStreamHandle>(stream_id);
  }();
#else
  AOTInductorStreamHandle stream_handle = nullptr;
#endif // __SIGRID_USE_GPU__
  AOTIProxyExecutorHandle proxy_executor_handle =
      reinterpret_cast<AOTIProxyExecutorHandle>(proxyExecutor_.get());

  auto result = runConstantFoldingFunc_(
      containerHandle_, use_inactive, stream_handle, proxy_executor_handle);

  TORCH_CHECK(
      result == AOTI_RUNTIME_SUCCESS, "Unable to run constant folding.");
}

void AOTInductorModelImpl::updateConstantBuffer(
    std::unordered_map<std::string, torch::Tensor*>&& constants,
    bool use_inactive,
    bool validate_full_update) {
  TORCH_CHECK(
      updateConstantBufferFunc_,
      "updateConstantBufferFunc_ was not loaded from .so");

  auto result = updateConstantBufferFunc_(
      containerHandle_,
      (AOTInductorConstantMapHandle)&constants,
      use_inactive,
      validate_full_update);
  TORCH_CHECK(
      result == AOTI_RUNTIME_SUCCESS, "Unable to update constant buffer");
}

void AOTInductorModelImpl::updateInactiveConstantBuffer(
    std::unordered_map<std::string, torch::Tensor*>&& constants) {
  TORCH_CHECK(
      updateInactiveConstantBufferFunc_,
      "updateInactiveConstantBufferFunc_ was not loaded from .so");

  auto result = updateInactiveConstantBufferFunc_(
      containerHandle_, (AOTInductorConstantMapHandle)&constants);
  TORCH_CHECK(
      result == AOTI_RUNTIME_SUCCESS,
      "Unable to update inactive constant buffer");
}

void AOTInductorModelImpl::swapConstantBuffers() {
  TORCH_CHECK(
      swapConstantBufferFunc_,
      "swapConstantBufferFunc_ was not loaded from .so");

  auto result = swapConstantBufferFunc_(containerHandle_);
  TORCH_CHECK(
      result == AOTI_RUNTIME_SUCCESS, "Unable to swap constant buffers");
}

thread_local std::unordered_map<std::string, std::string>
    AOTInductorModelImpl::lib_name_to_path_;

thread_local bool AOTInductorModelImpl::deserialize_pickled_model_{true};

thread_local std::optional<std::string> AOTInductorModelImpl::cubin_dir_;

thread_local std::unordered_map<std::string, std::string>
    AOTInductorModelImpl::extern_kernels_spec_name_to_path_;

void AOTInductorModelImpl::registerLibraryNameToPathMap(
    std::unordered_map<std::string, std::string> map) {
  std::ostringstream ss;
  ss << "{\n";
  for (const auto& [k, v] : map) {
    ss << "  " << k << " => " << v << ",\n";
  }
  ss << "}";

  LOG(INFO) << "Registering .so lib paths: " << ss.str();
  lib_name_to_path_ = std::move(map);
}

std::string AOTInductorModelImpl::getFullPathForLibraryName(
    const std::string& name) {
  auto path = lib_name_to_path_.find(name);
  std::ostringstream ss;
  ss << "{\n";
  for (const auto& [k, v] : lib_name_to_path_) {
    ss << "  " << k << " => " << v << ",\n";
  }
  if ((path == lib_name_to_path_.end()) ||
      (!std::filesystem::exists(path->second))) {
    for (const auto& lib_path : library_search_paths_) {
      std::string fullpath =
          lib_path + std::filesystem::path::preferred_separator + name;
      if (std::filesystem::exists(fullpath)) {
        return fullpath;
      }
      ss << "  searched for " << name << " at " << lib_path << ",\n";
    }
  }
  ss << "}";
  TORCH_CHECK(
      path != lib_name_to_path_.end(),
      "could not find full path for AOTInductor model .so named ",
      name,
      ". available paths: ",
      ss.str());
  return path->second;
}

void AOTInductorModelImpl::setCubinDir(std::optional<std::string> cubin_dir) {
  cubin_dir_ = cubin_dir;
}

std::optional<std::string> AOTInductorModelImpl::getCubinDir() {
  return cubin_dir_;
}

void AOTInductorModelImpl::registerExternKernelsSpecNameToPathMap(
    std::unordered_map<std::string, std::string> map) {
  std::ostringstream ss;
  ss << "{\n";
  for (const auto& [k, v] : map) {
    ss << "  " << k << " => " << v << ",\n";
  }
  ss << "}";

  LOG(INFO) << "Registering extern kernels spec paths: " << ss.str();
  extern_kernels_spec_name_to_path_ = std::move(map);
}

std::optional<std::string>
AOTInductorModelImpl::getFullPathForExternKernelsSpecName(
    const std::string& name) {
  auto it = extern_kernels_spec_name_to_path_.find(name);
  if (it == extern_kernels_spec_name_to_path_.end()) {
    LOG(INFO) << "Didn't find extern kernels spec file for " << name;
    return {};
  }
  if (!std::filesystem::exists(it->second)) {
    TORCH_CHECK(false, "Extern kernels spec file doesn't exist: ", it->second);
  }
  return it->second;
}

bool AOTInductorModelImpl::getDeserializePickledModel() {
  return deserialize_pickled_model_;
}

// Set thread local boolean to disable real loading from .so file
// for reusing the same module later on
void AOTInductorModelImpl::setDeserializePickledModel(
    bool deserializePickledModel) {
  deserialize_pickled_model_ = deserializePickledModel;
}

} // namespace torch::aot_inductor
