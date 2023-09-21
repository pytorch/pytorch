#include <torch/csrc/inductor/aot_runtime/interface.h>
#include <torch/csrc/inductor/aot_runtime/model_container.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, is_cpu, cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    // Array of raw AtenTensorHandle for output tensors. Handles will be stolen
    AtenTensorHandle* input_handles,
    size_t num_inputs,
    // Array of raw AtenTensorHandle for output tensors. Handles will be stolen
    AtenTensorHandle* output_handles,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle,
    const int64_t** ret_output_sizes,
    int64_t* ret_output_ndims) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);

  auto input_unique_handles =
      torch::aot_inductor::steal_from_raw_handles_to_raii_handles(input_handles, num_inputs);
  auto output_unique_handles =
      torch::aot_inductor::steal_from_raw_handles_to_raii_handles(output_handles, num_outputs);

  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);

  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<std::vector<int64_t>>* shapes;
    container->run(
        input_unique_handles,
        output_unique_handles,
        &shapes,
        stream,
        proxy_executor_handle);
    for (size_t i = 0; i < num_outputs; i++) {
      ret_output_sizes[i] = shapes->at(i).data();
      ret_output_ndims[i] = shapes->at(i).size();
    }
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_dtypes) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_dtypes = container->get_input_dtype(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_dtypes) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_dtypes = container->get_output_dtype(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetMaxInputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const int64_t** ret_input_sizes,
    int64_t* ret_input_ndim) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const std::vector<int64_t>& max_input_shape =
        container->max_input_shape(input_idx);
    *ret_input_sizes = max_input_shape.data();
    *ret_input_ndim = max_input_shape.size();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetMaxOutputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const int64_t** ret_output_sizes,
    int64_t* ret_output_ndim) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const std::vector<int64_t>& max_output_shape =
        container->max_output_shape(output_idx);
    *ret_output_sizes = max_output_shape.data();
    *ret_output_ndim = max_output_shape.size();
  })
}

} // extern "C"
