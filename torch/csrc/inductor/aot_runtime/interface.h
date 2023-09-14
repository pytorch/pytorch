#pragma once

#include <stddef.h>
#include <stdint.h>

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

struct AOTInductorProxyExecutorOpaque {};
using AOTInductorProxyExecutorHandle = AOTInductorProxyExecutorOpaque*;

extern "C" {
// Creates an AOTInductor model container. The parameter num_models
// specifies the number of model instances that may be run concurrently for
// the same input model.
AOTInductorError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models);

// Deletes the AOTInductor model container.
AOTInductorError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle);

// Runs the inference.
AOTInductorError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorTensorHandle inputs_handle,
    size_t num_inputs,
    AOTInductorTensorHandle outputs_handle,
    size_t num_outputs,
    AOTInductorParamShape* output_shapes,
    AOTInductorStreamHandle stream_handle,
    AOTInductorProxyExecutorHandle proxy_executor_handle);

// Retrieves the number of inputs for the model.
AOTInductorError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_inputs_out);

// Retrieves the input name at the given index.
AOTInductorError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_name_out);

// Retrieves the input dtype at the given index.
AOTInductorError AOTInductorModelContainerGetInputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** input_name_out);

// Retrieves the number of outputs for the model.
AOTInductorError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_outputs_out);

// Retrieves the output name at the given index.
AOTInductorError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** output_name_out);

// Retrieves the output dtype at the given index.
AOTInductorError AOTInductorModelContainerGetOutputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** output_name_out);

// Retieves the input shape with the maximum dimension size for each dimension.
AOTInductorError AOTInductorModelContainerGetMaxInputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    AOTInductorParamShape* input_shape);

// Retieves the output shape with the maximum dimension size for each dimension.
AOTInductorError AOTInductorModelContainerGetMaxOutputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    AOTInductorParamShape* output_shape);

} // extern "C"
