#include <torch/csrc/inductor/aot_inductor_interface.h>
#include <torch/csrc/inductor/aot_inductor_model_container.h>
#include <iostream>
#include <torch/csrc/inductor/proxy_executor.h>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                                  \
  try {                                                                       \
    __VA_ARGS__                                                               \
  } catch (const std::exception& e) {                                         \
    std::cerr << "Error: " << e.what() << std::endl;                          \
    return AOTInductorError::Failure;                                         \
  } catch (...) {                                                             \
    std::cerr << "Unknown exception occurred." << std::endl;                  \
    return AOTInductorError::Failure;                                         \
  }                                                                           \
  return AOTInductorError::Success;

extern "C" {

AOTInductorError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models) {
  if (num_models == 0) {
    LOG(ERROR) << "num_models must be positive, but got 0";
    return AOTInductorError::Failure;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        new torch::aot_inductor::AOTInductorModelContainer(num_models);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTInductorError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTInductorError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorTensorHandle inputs_handle,
    size_t num_inputs,
    AOTInductorTensorHandle outputs_handle,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTInductorProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);

  const auto* inputs = reinterpret_cast<const at::Tensor*>(inputs_handle);
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    input_tensors.push_back(inputs[i]);
  }

  auto* outputs = reinterpret_cast<at::Tensor*>(outputs_handle);
  std::vector<at::Tensor> output_tensors;
  output_tensors.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    output_tensors.push_back(outputs[i]);
  }

  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);

  torch::aot_inductor::ProxyExecutor* proxy_executor = reinterpret_cast<torch::aot_inductor::ProxyExecutor*>(proxy_executor_handle);

  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { container->run(input_tensors, output_tensors, stream, proxy_executor); })
}

AOTInductorError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_inputs_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *num_inputs_out = container->num_inputs(); })
}

AOTInductorError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_name_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *input_name_out = container->input_name(input_idx); })
}

AOTInductorError AOTInductorModelContainerGetInputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_dtype_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *input_dtype_out = container->get_input_dtype(input_idx); })
}

AOTInductorError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_outputs_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *num_outputs_out = container->num_outputs(); })
}

AOTInductorError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** output_name_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *output_name_out = container->output_name(output_idx); })
}

AOTInductorError AOTInductorModelContainerGetOutputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** output_dtype_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *output_dtype_out = container->get_output_dtype(output_idx); })
}

AOTInductorError AOTInductorModelContainerGetMaxInputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    AOTInductorParamShape* input_shape) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const std::vector<int64_t>& max_input_shape =
        container->max_input_shape(input_idx);
    *input_shape =
        AOTInductorParamShape(max_input_shape.data(), max_input_shape.size());
  })
}

AOTInductorError AOTInductorModelContainerGetMaxOutputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    AOTInductorParamShape* output_shape) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const std::vector<int64_t>& max_output_shape =
        container->max_output_shape(output_idx);
    *output_shape =
        AOTInductorParamShape(max_output_shape.data(), max_output_shape.size());
  })
}

} // extern "C"
