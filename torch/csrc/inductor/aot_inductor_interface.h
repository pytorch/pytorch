#pragma once

#include <stddef.h>
#include <stdint.h>

#include <torch/csrc/inductor/aot_inductor_model_container.h>

#ifdef __GNUC__
#define AOT_INDUCTOR_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
#define AOT_INDUCTOR_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOT_INDUCTOR_EXPORT
#endif // _WIN32
#endif // __GNUC__

enum class AOTInductorError : int {
  Success = 0,
  Failure = 1,
};

#define AOT_INDUCTOR_ERROR_CHECK(call)                                    \
  if ((call) != AOTInductorError::Success) {                              \
    throw std::runtime_error(                                             \
        std::string(#call " API call failed at ") + __FILE__ + ", line" + \
        std::to_string(__LINE__));                                        \
  }

// The shape representation passed through the C interfaces.
struct AOTInductorParamShape {
  AOTInductorParamShape() : shape_data(nullptr), ndim(0) {}
  AOTInductorParamShape(const int64_t* data, int64_t n)
      : shape_data(data), ndim(n) {}

  const int64_t* shape_data;
  int64_t ndim;
};

struct AOTInductorModelContainerOpaque {};
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

struct AOTInductorStreamOpaque {};
using AOTInductorStreamHandle = AOTInductorStreamOpaque*;

struct AOTInductorTensorOpaque {};
using AOTInductorTensorHandle = AOTInductorTensorOpaque*;

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)     \
  try {                                          \
    __VA_ARGS__                                  \
  } catch (const std::exception& e) {            \
    LOG(ERROR) << "Error: " << e.what();         \
    return AOTInductorError::Failure;            \
  } catch (...) {                                \
    LOG(ERROR) << "Unknown exception occurred."; \
    return AOTInductorError::Failure;            \
  }                                              \
  return AOTInductorError::Success;

extern "C" {
// Creates an AOTInductor model container. The parameter num_models
// specifies the number of model instances that may be run concurrently for
// the same input model.
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

// Deletes the AOTInductor model container.
AOTInductorError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

// Runs the inference.
AOTInductorError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorTensorHandle inputs_handle,
    size_t num_inputs,
    AOTInductorTensorHandle outputs_handle,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle) {
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
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { container->run(input_tensors, output_tensors, stream); })
}

// Retrieves the number of inputs for the model.
AOTInductorError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_inputs_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *num_inputs_out = container->num_inputs(); })
}

// Retrieves the input name at the given index.
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

// Retrieves the number of outputs for the model.
AOTInductorError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_outputs_out) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *num_outputs_out = container->num_outputs(); })
}

// Retrieves the output name at the given index.
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

// Retieves the input shape with the maximum dimension size for each dimension.
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

// Retieves the output shape with the maximum dimension size for each dimension.
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
