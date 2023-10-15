#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <ATen/DynamicLibrary.h>

#include <torch/csrc/inductor/aoti_model_runner.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

namespace torch::inductor {

AOTIModelRunner::AOTIModelRunner(
    const char* model_path,
    const ConstantMap& const_map) {
  model_so_ = std::make_unique<at::DynamicLibrary>(model_path);
  TORCH_CHECK(model_so_, "Failed to load model: ", model_path);
  create_func_ = reinterpret_cast<decltype(create_func_)>(
      model_so_->sym("AOTInductorModelCreate"));
  delete_func_ = reinterpret_cast<decltype(delete_func_)>(
      model_so_->sym("AOTInductorModelDelete"));
  get_num_outputs_func_ = reinterpret_cast<decltype(get_num_outputs_func_)>(
      model_so_->sym("AOTInductorModelGetNumOutputs"));
  run_func_ = reinterpret_cast<decltype(run_func_)>(
      model_so_->sym("AOTInductorModelRun"));
  update_constants_func_ = reinterpret_cast<decltype(update_constants_func_)>(
      model_so_->sym("AOTInductorModelUpdateConstants"));
  update_constants_map_func_ =
      reinterpret_cast<decltype(update_constants_map_func_)>(
          model_so_->sym("AOTInductorModelUpdateConstantsMap"));

  AOTI_RUNTIME_ERROR_CODE_CHECK(
      create_func_(&model_handle_, (AOTInductorConstantMapHandle)&const_map));
}

AOTIModelRunner::~AOTIModelRunner() {
  AOTIRuntimeError result = delete_func_(model_handle_);
  TORCH_CHECK(result == AOTI_RUNTIME_SUCCESS, "AOTInductorModelDelete failed");
}

std::vector<at::Tensor> AOTIModelRunner::run(std::vector<at::Tensor> inputs) {
  auto input_handles =
      torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(inputs);

  // For outputs, we only allocate a vector to hold returned tensor handles,
  // not allocating the actual output tensor storage here
  size_t num_outputs = 0;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_outputs_func_(model_handle_, &num_outputs));
  std::vector<AtenTensorHandle> output_handles(num_outputs);

  AOTI_RUNTIME_ERROR_CODE_CHECK(
      run_func_(model_handle_, input_handles.data(), output_handles.data()));

  return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
      output_handles.data(), output_handles.size());
}

void AOTIModelRunner::update_constants(const ConstantMap& const_map) {
  AOTI_RUNTIME_ERROR_CODE_CHECK(update_constants_func_(
      model_handle_, (AOTInductorConstantMapHandle)&const_map));
}

void AOTIModelRunner::update_constants_map(const ConstantMap& const_map) {
  AOTI_RUNTIME_ERROR_CODE_CHECK(update_constants_map_func_(
      model_handle_, (AOTInductorConstantMapHandle)&const_map));
}

} // namespace torch::inductor
#endif
