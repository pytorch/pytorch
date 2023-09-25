#pragma once

#include <stddef.h>
#include <stdint.h>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aot_runtime/.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __GNUC__
#define AOT_INDUCTOR_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
#define AOT_INDUCTOR_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOT_INDUCTOR_EXPORT
#endif // _WIN32
#endif // __GNUC__

using AOTIRuntimeError = int32_t;
#define AOTI_RUNTIME_SUCCESS 0
#define AOTI_RUNTIME_FAILURE 1

#define AOTI_RUNTIME_ERROR_CODE_CHECK(call)                                \
  if ((call) != AOTI_RUNTIME_SUCCESS) {                                    \
    throw std::runtime_error(                                              \
        std::string(#call " API call failed at ") + __FILE__ + ", line " + \
        std::to_string(__LINE__));                                         \
  }

extern "C" {
struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

struct AOTInductorStreamOpaque;
using AOTInductorStreamHandle = AOTInductorStreamOpaque*;

// Creates an AOTInductor model container. The parameter num_models
// specifies the number of model instances that may be run concurrently for
// the same input model.
AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir);

// Deletes the AOTInductor model container.
AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle);

// Runs the inference.
AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// Retrieves the number of inputs for the model.
AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs);

// Retrieves the input name at the given index.
AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names);

// Retrieves the input dtype at the given index.
AOTIRuntimeError AOTInductorModelContainerGetInputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_dtypes);

// Retrieves the number of outputs for the model.
AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs);

// Retrieves the output name at the given index.
AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names);

// Retrieves the output dtype at the given index.
AOTIRuntimeError AOTInductorModelContainerGetOutputDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_dtypes);

// Retieves the input shape with the maximum dimension size for each dimension.
AOTIRuntimeError AOTInductorModelContainerGetMaxInputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const int64_t** ret_input_sizes,
    int64_t* ret_input_ndim);

// Retieves the output shape with the maximum dimension size for each dimension.
AOTIRuntimeError AOTInductorModelContainerGetMaxOutputShape(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const int64_t** ret_output_sizes,
    int64_t* ret_output_ndim);

} // extern "C"
