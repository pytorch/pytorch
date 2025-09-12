#pragma once

#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cstdint>

// This header defines a prototype stable C API for certain Module
// functionality. It is inspired by:
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/c/shim.h
//
// NOTE: We are not actually guaranteeing ABI stability on this API yet as
// it is in a highly experimental state.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The type of a StableIValue.
 */
enum class StableIValueTag : uint32_t {
  None = 0,
  Tensor = 1,
  Double = 2,
  Int = 3,
  Bool = 4,
};

/**
 * A wrapper containing StableIValue which is opaque and its actual type.
 *
 * StableIValue is typically used when interacting with the dispatcher.
 * There the type is known ahead of time as the operators have schemas,
 * and the value is guaranteed by the caller to be of the correct type.
 *
 * Here the value comes from user space. While we could technically
 * infer the expected type from the module, it is not clear how to
 * validate the user args which seems like a massive footgun. So for
 * now, we have the user manually specify the type. This struct is likely
 * to change as we iterate on the API.
 */
struct TypedStableIValue {
  StableIValue val;
  StableIValueTag tag;
};

struct ModuleOpaque;
using ModuleHandle = ModuleOpaque*;

AOTI_TORCH_EXPORT AOTITorchError experimental_torch_load_module_from_file(
    const char* package_path,
    uint64_t package_path_len,
    const char* model_name,
    uint64_t model_name_len,
    ModuleHandle* ret_value);

AOTI_TORCH_EXPORT AOTITorchError
experimental_torch_delete_module_object(ModuleHandle handle);

AOTI_TORCH_EXPORT AOTITorchError
experimental_torch_module_num_outputs(ModuleHandle handle, uint64_t* ret_value);

AOTI_TORCH_EXPORT AOTITorchError experimental_torch_module_forward_flattened(
    ModuleHandle handle,
    TypedStableIValue* args,
    uint64_t num_args,
    TypedStableIValue* ret_values,
    uint64_t num_outputs);

#ifdef __cplusplus
} // extern "C"
#endif
