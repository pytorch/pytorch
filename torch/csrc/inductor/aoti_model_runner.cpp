#include <dlfcn.h>
#include <iostream>

#include <torch/csrc/inductor/aoti_model_runner.h>

#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

namespace torch {
namespace inductor {

AOTIModelRunner::AOTIModelRunner(
    const char* model_so_path,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir)
    : model_so_{dlopen(model_so_path, RTLD_NOW | RTLD_LOCAL)} {
  TORCH_CHECK(model_so_, "Failure loading .so: ", dlerror());

#define LOAD_SYMBOL(var, name_str)                                   \
  var = reinterpret_cast<decltype(var)>(dlsym(model_so_, name_str)); \
  TORCH_CHECK(var, "could not dlsym " name_str);

  LOAD_SYMBOL(create_func_, "AOTInductorModelContainerCreate");
  LOAD_SYMBOL(delete_func_, "AOTInductorModelContainerDelete");
  LOAD_SYMBOL(get_num_outputs_func_, "AOTInductorModelContainerGetNumOutputs");
  LOAD_SYMBOL(run_func_, "AOTInductorModelContainerRun");
#undef LOAD_SYMBOL

  AOTI_RUNTIME_ERROR_CODE_CHECK(
      create_func_(&container_handle_, num_models, is_cpu, cubin_dir));
}

AOTIModelRunner::~AOTIModelRunner() {
  if (delete_func_(container_handle_) != AOTI_RUNTIME_SUCCESS) {
    std::cerr << "Failed to delete model container" << std::endl;
  }
  if (dlclose(model_so_) != 0) {
    std::cerr << "Failed to close shared lib: " << dlerror() << std::endl;
  }
}

std::vector<at::Tensor> AOTIModelRunner::run(
    std::vector<at::Tensor> inputs,
    AOTInductorStreamHandle cuda_stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
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
      proxy_executor_handle));

  return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
      output_handles.data(), output_handles.size());
}

} // namespace inductor
} // namespace torch
