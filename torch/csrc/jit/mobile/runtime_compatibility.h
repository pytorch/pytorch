#pragma once

#include <memory>
#include <unordered_map>

namespace torch {
namespace jit {

static const int NO_SCHEMA = -1;

struct OperatorInfo {
  int num_schema_args = NO_SCHEMA;
};

TORCH_API uint64_t _get_runtime_bytecode_version();

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_runtime_ops_and_info();

} // namespace jit
} // namespace torch
