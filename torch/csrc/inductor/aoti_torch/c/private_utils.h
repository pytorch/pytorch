#pragma once

// NOTE: This header is NOT ABI stable and may change between PyTorch versions.
// Unlike torch/csrc/inductor/aoti_torch/c/shim.h, which provides ABI stability
// guarantees, functions in this header are subject to change and should be
// considered internal implementation details.

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus

// Convert from IValue to StableIValue with ABI version support
AOTI_TORCH_EXPORT StableIValue from_ivalue(
    const c10::TypePtr& type,
    const c10::IValue& ivalue,
    uint64_t extension_abi_version);

// Convert from StableIValue to IValue with ABI version support
AOTI_TORCH_EXPORT c10::IValue to_ivalue(
    const c10::TypePtr& type,
    const StableIValue stable_ivalue,
    uint64_t extension_abi_version);

// Schema adapter registration function
AOTI_TORCH_EXPORT AOTITorchError register_schema_adapter(
    const char* op_name,
    uint64_t max_version,
    void* adapter_fn); // Use void* to avoid namespace conflicts

#endif // __cplusplus
