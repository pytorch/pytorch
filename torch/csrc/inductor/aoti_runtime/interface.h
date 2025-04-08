#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>

extern "C" {
struct AOTInductorModelOpaque;
using AOTInductorModelHandle = AOTInductorModelOpaque*;

struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

struct AOTInductorStreamOpaque;
using AOTInductorStreamHandle = AOTInductorStreamOpaque*;

struct AOTInductorConstantMap;
using AOTInductorConstantMapHandle = AOTInductorConstantMap*;

// TODO: Deprecate this API. This was kept for BC compatibility.
// Please use AOTInductorModelContainerCreateWithDevice instead.
AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir);

// Creates an AOTInductor model container. The parameter num_models
// specifies the number of model instances that may be run concurrently for
// the same input model.
// `device_str` MUST NOT be nullptr. It must be a valid device string, e.g.
// "cpu", "cuda", "cuda:0", etc. If the device index is not specified for CUDA
// device, runtime will use the device index returned by
// "cudaGetDevice(&device_idx)"
AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
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

// Single-threaded variant of previous.
AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
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

// Retrieves the number of constants for the model.
AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

// Retrieves a constant's name.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name);

// Retrieves a constant's original FQN.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn);

// Retrieves whether a constant is from folded.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded);

// Retrieves the inductor constant type.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type);

// Retrieves a constant's dtype.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype);

// Setup the constant buffer in model container with provided ConstantMap
// use_inactive should be set as true if the inactive buffer is to be updated.
// validate_full_update checks if all constants are included in the ConstantMap
AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update);

// Setup the inactive constant buffer in model container with provided
// ConstantMap
AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Run constant folding on constant buffer.
AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// Swap the constant buffer being used to the inactive one.
AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle);

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
