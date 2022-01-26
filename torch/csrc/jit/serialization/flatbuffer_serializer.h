#pragma once

#include <ATen/core/qualified_name.h>
#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <torch/csrc/jit/serialization/export.h>

#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h> // NOLINT

namespace torch {
namespace jit {

TORCH_API void save_mobile_module(
    const mobile::Module& module,
    const std::string& filename,
    const bool save_mobile_debug_info = false);
TORCH_API flatbuffers::DetachedBuffer save_mobile_module_to_bytes(
    const mobile::Module& module,
    const bool save_mobile_debug_info = false);


// This caches serialized inlined callstack ptr, since many
// InlinedCallStackPtr can refer to the same one.
ska::flat_hash_map<InlinedCallStackPtr, c10::IValue>
    serialized_inlined_callstack_;

ska::flat_hash_map<std::string, c10::IValue> serialized_module_instance_info_;

} // namespace jit
} // namespace torch
