#pragma once

#include <c10/macros/Export.h>
#include <optional>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace torch::jit {

// Struct storing metadata of an operator that can be useful for versioning
struct OperatorInfo {
  // The number of arguments within the schema of the op
  std::optional<int> num_schema_args;
};

struct RuntimeCompatibilityInfo {
  std::pair<uint64_t, uint64_t> min_max_supported_bytecode_version;
  std::unordered_map<std::string, OperatorInfo> operator_info;
  std::unordered_set<std::string> supported_types;
  std::pair<uint64_t, uint64_t> min_max_supported_operator_versions;

  // Factory Method
  static TORCH_API RuntimeCompatibilityInfo get();
};

TORCH_API uint64_t _get_runtime_bytecode_version();

TORCH_API std::pair<uint64_t, uint64_t> _get_runtime_bytecode_min_max_versions();

TORCH_API std::pair<uint64_t, uint64_t>
_get_runtime_operators_min_max_versions();

TORCH_API std::unordered_map<std::string, OperatorInfo>
_get_runtime_ops_and_info();

TORCH_API std::unordered_set<std::string> _get_mobile_supported_types();

TORCH_API std::unordered_set<std::string> _get_loaded_custom_classes();

} // namespace torch::jit
