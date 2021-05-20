#pragma once

#include <memory>
#include <unordered_map>

namespace torch {
namespace jit {

struct OperatorInfo {
  c10::optional<int> num_schema_args;
};

TORCH_API uint64_t _get_runtime_bytecode_version();

TORCH_API std::unordered_map<std::string, OperatorInfo>
_get_runtime_ops_and_info();

} // namespace jit
} // namespace torch
