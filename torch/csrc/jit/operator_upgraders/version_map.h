#pragma once
#include <c10/macros/Export.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

struct UpgraderEntry {
  int bumped_at_version;
  std::string upgrader_name;
  std::string old_schema;
};

// Toggle the behaviour of calculating version for the module.
// If this is true, we calculate solely based on upgraders
// If this is false, we calculate it based on historic per op version map
TORCH_API void calculate_package_version_based_on_upgraders(bool val);

TORCH_API bool get_version_calculator_flag();

TORCH_API const std::unordered_map<std::string, std::vector<UpgraderEntry>>&
get_operator_version_map();

TORCH_API void test_only_add_entry(
    const std::string& op_name,
    UpgraderEntry entry);

TORCH_API void test_only_remove_entry(const std::string& op_name);

TORCH_API void test_only_reset_flag();

} // namespace jit
} // namespace torch
