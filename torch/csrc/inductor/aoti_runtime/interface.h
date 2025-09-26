#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>

#ifdef _WIN32
/*
On Windows, we need to explicit declaration for export APIs. And because the
package loader call these API via GetProcAddress(ldsym on Linux), we can ignore
the import case.
*/
#define AOTI_API __declspec(dllexport)
#else
#define AOTI_API __attribute__((__visibility__("default")))
#endif

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
AOTI_API AOTIRuntimeError AOTInductorModelContainerCreate(
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
AOTI_API AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

// Deletes the AOTInductor model container.
AOTI_API AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle);

// Runs the inference.
AOTI_API AOTIRuntimeError AOTInductorModelContainerRun(
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
AOTI_API AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
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
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

// Retrieves a constant's name.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name);

// Retrieves a constant's original FQN.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn);

// Retrieves whether a constant is from folded.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded);

// Retrieves the inductor constant type.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type);

// Retrieves a constant's dtype.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype);

// Retrieves a constant's data size.
// idx is the index of the internal's constants.
// Need idx < num_constants from AOTInductorModelContainerGetNumConstants
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    size_t* data_size);

// Extract the constants that is being used in the container.
AOTI_API AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive);

// Setup the constant buffer in model container with provided ConstantMap.
// The ConstantMap is user managed, and the user would retain ownership.
AOTI_API AOTIRuntimeError
AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update);

// Setup the constant buffer in model container with provided ConstantMap
// use_inactive should be set as true if the inactive buffer is to be updated.
// validate_full_update checks if all constants are included in the ConstantMap
AOTI_API AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update);

// Setup the inactive constant buffer in model container with provided
// ConstantMap
AOTI_API AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Free the inactive constant buffer in model container.
AOTI_API AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle);

// Run constant folding on constant buffer.
AOTI_API AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// Swap the constant buffer being used to the inactive one.
AOTI_API AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle);

// Retrieves the number of inputs for the model.
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs);

// Retrieves the input name at the given index.
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names);

// Retrieves the number of outputs for the model.
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs);

// Retrieves the output name at the given index.
AOTI_API AOTIRuntimeError AOTInductorModelContainerGetOutputName(
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
AOTI_API AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Run an AOTInductorModel (see AOTInductorModelCreate for when one should use
// this function versus AOTInductorModelContainerRun).
AOTI_API AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles);

// Replace AOTInductorModel's constant map. Note it doesn't handle concurrency
// so be sure to handle ordering if AOTInductorModelRun is ran concurrently.
AOTI_API AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Delete an AOTInductorModel created by AOTInductorModelCreate.
AOTI_API AOTIRuntimeError
AOTInductorModelDelete(AOTInductorModelHandle model_handle);

AOTI_API AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs);

AOTI_API AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec);

} // extern "C"
