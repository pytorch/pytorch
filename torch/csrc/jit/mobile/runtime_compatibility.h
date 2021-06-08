#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Optional.h>

#include <memory>
#include <unordered_map>

namespace torch {
namespace jit {

// Struct storing metadata of an operator that can be useful for versioning
struct OperatorInfo {
  // The number of arguments within the schema of the op
  c10::optional<int> num_schema_args;
};

TORCH_API uint64_t _get_runtime_bytecode_version();

TORCH_API std::unordered_map<std::string, OperatorInfo>
_get_runtime_ops_and_info();

} // namespace jit
} // namespace torch
