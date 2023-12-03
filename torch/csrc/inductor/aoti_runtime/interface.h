#pragma once

#include <stddef.h>
#include <stdint.h>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
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
struct AOTInductorModelOpaque;
using AOTInductorModelHandle = AOTInductorModelOpaque*;

struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

struct AOTInductorStreamOpaque;
using AOTInductorStreamHandle = AOTInductorStreamOpaque*;

struct AOTInductorConstantMap;
using AOTInductorConstantMapHandle = AOTInductorConstantMap*;

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

// Retrieves the number of outputs for the model.
AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs);

// Retrieves the output name at the given index.
AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names);

// Creates an AOTInductorModel instance.  This is a thin and light wrapper
// around the compiled model; it doesn't handle concurrency, queueing, device
// management, etc.  Use this if bare-metal performance is needed and you are
// willing to handle other "management" aspects yourself.
//
// constant_map_handle is an opaque type to satisfy the C ABI.  It should be a
// std::unordered_map<std::string, at::Tensor*>*.
AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Run an AOTInductorModel (see AOTInductorModelCreate for when one should use
// this function versus AOTInductorModelContainerRun).
AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles);

// Replace AOTInductorModel's constant map. Note it doesn't handle concurrency
// so be sure to handle ordering if AOTInductorModelRun is ran concurrently.
AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Delete an AOTInductorModel created by AOTInductorModelCreate.
AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle);

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs);

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec);

} // extern "C"
