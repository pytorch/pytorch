#include <torch/csrc/inductor/aot_inductor_interface.h>
#include <torch/csrc/inductor/aot_inductor_model_container.h>
#include <torch/csrc/inductor/aot_inductor_tensor.h>
#include <stdexcept>
#include <vector>
#include <iostream>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)     \
  try {                                          \
    __VA_ARGS__                                  \
  } catch (const std::exception& e) {            \
    std::cerr << "Error: " << e.what();         \
    return AOTInductorError::Failure;            \
  } catch (...) {                                \
    std::cerr << "Unknown exception occurred."; \
    return AOTInductorError::Failure;            \
  }                                              \
  return AOTInductorError::Success;

extern "C" {

AOTInductorError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models) {
  if (num_models == 0) {
    std::cerr << "num_models must be positive, but got 0";
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
    AOTInductorStreamHandle stream_handle) {
  aot_inductor_initialize();

  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);

  auto inputs = reinterpret_cast<AotInductorTensor*>(inputs_handle);
  std::vector<AotInductorTensor> input_tensors(inputs, inputs+num_inputs);

  auto outputs = reinterpret_cast<AotInductorTensor*>(outputs_handle);
  std::vector<AotInductorTensor> output_tensors(outputs, outputs+num_outputs);

  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { container->run(input_tensors, output_tensors, stream);})

  aot_inductor_destroy();
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
