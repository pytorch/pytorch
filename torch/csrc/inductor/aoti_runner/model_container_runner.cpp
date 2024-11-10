#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <ATen/DynamicLibrary.h>

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

#ifndef _WIN32
#include <sys/stat.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace {
bool file_exists(std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc {};
  return lstat(path.c_str(), &rc) == 0;
#endif
}
} // namespace

namespace torch::inductor {

AOTIModelContainerRunner::AOTIModelContainerRunner(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir) {
  model_so_ = std::make_unique<at::DynamicLibrary>(model_so_path.c_str());
  TORCH_CHECK(model_so_, "Failed to load model: ", model_so_path);
  create_func_ = reinterpret_cast<decltype(create_func_)>(
      model_so_->sym("AOTInductorModelContainerCreateWithDevice"));
  delete_func_ = reinterpret_cast<decltype(delete_func_)>(
      model_so_->sym("AOTInductorModelContainerDelete"));
  get_num_outputs_func_ = reinterpret_cast<decltype(get_num_outputs_func_)>(
      model_so_->sym("AOTInductorModelContainerGetNumOutputs"));
  run_func_ = reinterpret_cast<decltype(run_func_)>(
      model_so_->sym("AOTInductorModelContainerRun"));
  get_num_constants_func_ = reinterpret_cast<decltype(get_num_constants_func_)>(
      model_so_->sym("AOTInductorModelContainerGetNumConstants"));
  get_constant_name_func_ = reinterpret_cast<decltype(get_constant_name_func_)>(
      model_so_->sym("AOTInductorModelContainerGetConstantName"));
  get_constant_original_fqn_func_ =
      reinterpret_cast<decltype(get_constant_original_fqn_func_)>(
          model_so_->sym("AOTInductorModelContainerGetConstantOriginalFQN"));
  get_constant_dtype_func_ =
      reinterpret_cast<decltype(get_constant_dtype_func_)>(
          model_so_->sym("AOTInductorModelContainerGetConstantDtype"));
  update_constant_buffer_func_ =
      reinterpret_cast<decltype(update_constant_buffer_func_)>(
          model_so_->sym("AOTInductorModelContainerUpdateConstantBuffer"));
  update_inactive_constant_buffer_func_ =
      reinterpret_cast<decltype(update_inactive_constant_buffer_func_)>(
          model_so_->sym(
              "AOTInductorModelContainerUpdateInactiveConstantBuffer"));
  run_const_fold_func_ = reinterpret_cast<decltype(run_const_fold_func_)>(
      model_so_->sym("AOTInductorModelContainerRunConstantFolding"));
  swap_constant_buffer_func_ =
      reinterpret_cast<decltype(swap_constant_buffer_func_)>(
          model_so_->sym("AOTInductorModelContainerSwapConstantBuffer"));
  get_call_spec_func_ = reinterpret_cast<decltype(get_call_spec_func_)>(
      model_so_->sym("AOTInductorModelContainerGetCallSpec"));

  // Hack to find the json file name from the model so file
  size_t lastindex = model_so_path.find_last_of('.');
  std::string json_filename = model_so_path.substr(0, lastindex) + ".json";

  if (file_exists(json_filename)) {
    proxy_executor_ = std::make_unique<torch::aot_inductor::OSSProxyExecutor>(
        json_filename, device_str == "cpu");
    proxy_executor_handle_ =
        reinterpret_cast<AOTIProxyExecutorHandle>(proxy_executor_.get());
  }

  AOTI_RUNTIME_ERROR_CODE_CHECK(create_func_(
      &container_handle_,
      num_models,
      device_str.c_str(),
      cubin_dir.empty() ? nullptr : cubin_dir.c_str()));
}

AOTIModelContainerRunner::~AOTIModelContainerRunner() {
  AOTIRuntimeError result = delete_func_(container_handle_);
  TORCH_CHECK(
      result == AOTI_RUNTIME_SUCCESS, "AOTInductorModelContainerDelete failed");
}

std::vector<at::Tensor> AOTIModelContainerRunner::run(
    const std::vector<at::Tensor>& inputs,
    AOTInductorStreamHandle cuda_stream_handle) {
  auto input_handles =
      torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(inputs);

  // For outputs, we only allocate a vector to hold returned tensor handles,
  // not allocating the actual output tensor storage here
  size_t num_outputs = 0;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_outputs_func_(container_handle_, &num_outputs));
  std::vector<AtenTensorHandle> output_handles(num_outputs);

  AOTI_RUNTIME_ERROR_CODE_CHECK(run_func_(
      container_handle_,
      input_handles.data(),
      input_handles.size(),
      output_handles.data(),
      output_handles.size(),
      cuda_stream_handle,
      proxy_executor_handle_));

  return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
      output_handles.data(), output_handles.size());
}

std::unordered_map<std::string, std::string> AOTIModelContainerRunner::
    getConstantNamesToOriginalFQNs() const {
  std::unordered_map<std::string, std::string> result;
  size_t num_constants{0};
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_constants_func_(container_handle_, &num_constants));
  for (size_t i = 0; i < num_constants; ++i) {
    const char* name{nullptr};
    const char* original_fqn{nullptr};
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_name_func_(container_handle_, i, &name));
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_original_fqn_func_(container_handle_, i, &original_fqn));
    result.emplace(name, original_fqn);
  }
  return result;
}

std::unordered_map<std::string, int32_t> AOTIModelContainerRunner::
    getConstantNamesToDtypes() const {
  std::unordered_map<std::string, int32_t> result;
  size_t num_constants{0};
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_constants_func_(container_handle_, &num_constants));
  for (size_t i = 0; i < num_constants; ++i) {
    const char* name{nullptr};
    int32_t dtype{0};
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_name_func_(container_handle_, i, &name));
    AOTI_RUNTIME_ERROR_CODE_CHECK(
        get_constant_dtype_func_(container_handle_, i, &dtype));
    result.emplace(name, dtype);
  }
  return result;
}

void AOTIModelContainerRunner::update_constant_buffer(
    const TensorConstantMap& const_map,
    bool use_inactive,
    bool check_full_update) {
  AOTI_RUNTIME_ERROR_CODE_CHECK(update_constant_buffer_func_(
      container_handle_,
      (AOTInductorConstantMapHandle)&const_map,
      use_inactive,
      check_full_update));
}

void AOTIModelContainerRunner::update_inactive_constant_buffer(
    const TensorConstantMap& const_map) {
  AOTI_RUNTIME_ERROR_CODE_CHECK(update_inactive_constant_buffer_func_(
      container_handle_, (AOTInductorConstantMapHandle)&const_map));
}

void AOTIModelContainerRunner::run_const_fold(
    bool use_inactive,
    AOTInductorStreamHandle cuda_stream_handle) {
  AOTI_RUNTIME_ERROR_CODE_CHECK(run_const_fold_func_(
      container_handle_,
      use_inactive,
      cuda_stream_handle,
      proxy_executor_handle_));
}

void AOTIModelContainerRunner::swap_constant_buffer() {
  AOTI_RUNTIME_ERROR_CODE_CHECK(swap_constant_buffer_func_(container_handle_));
}

std::vector<std::string> AOTIModelContainerRunner::get_call_spec() {
  const char* in_spec = nullptr;
  const char* out_spec = nullptr;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_call_spec_func_(container_handle_, &in_spec, &out_spec));
  return {in_spec, out_spec};
}

std::unordered_map<std::string, CreateAOTIModelRunnerFunc>&
getAOTIModelRunnerRegistry() {
  static std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      aoti_model_runner_registry_;
  return aoti_model_runner_registry_;
}

} // namespace torch::inductor
#endif
