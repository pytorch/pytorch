#pragma once

#include <cstdint>
#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>


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
 */
struct TypedStableIValue{
    StableIValue val;
    StableIValueTag tag;
};

struct ModuleOpaque;
using ModuleHandle = ModuleOpaque*;

AOTI_TORCH_EXPORT AOTITorchError
experimental_torch_load_module_from_file(
  const char* package_path,
  uint64_t package_path_len,
  const char* model_name,
  uint64_t model_name_len,
  ModuleHandle* ret_value);

AOTI_TORCH_EXPORT AOTITorchError 
experimental_torch_delete_module_object(ModuleHandle handle);

AOTI_TORCH_EXPORT AOTITorchError 
experimental_torch_module_num_outputs(
  ModuleHandle handle,
  uint64_t* ret_value);

AOTI_TORCH_EXPORT AOTITorchError 
experimental_torch_module_forward_flattened(
  ModuleHandle handle,
  const TypedStableIValue* args,
  uint64_t num_args,
  TypedStableIValue* ret_values,
  uint64_t num_outputs);

#ifdef __cplusplus
} // extern "C"
#endif
