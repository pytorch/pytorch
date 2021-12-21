#include <torch/csrc/jit/operator_upgraders/version_map.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

// Main entry point for all operators that have valid upgraders.
// Note for developers: The list of upgraders need to be SORTED
// by the version number where the upgrader is registered.
static std::unordered_map<std::string, std::vector<UpgraderEntry>> operatorVersionMap(
    {{"aten::div.Tensor",
      {{4,
        "div_Tensor_0_3",
        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},
     {"aten::div.Scalar",
      {{4,
        "div_Scalar_0_3",
        "aten::div.Scalar(Tensor self, Scalar other) -> Tensor"}}},
     {"aten::div.out",
      {{4,
        "div_out_0_3",
        "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"}}},
     {"aten::div_.Tensor",
      {{4,
        "div__Tensor_0_3",
        "aten::div_.Tensor(Tensor(a!), Tensor other) -> Tensor(a!)"}}},
     {"aten::div_.Scalar",
      {{4,
        "div__Scalar_0_3",
        "aten::div_.Scalar(Tensor(a!), Tensor other) -> Tensor(a!)"}}},
     {"aten::full",
      {{5,
        "full_0_4",
        "aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"aten::full.out",
      {{5,
        "full_out_0_4",
        "aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"}}}});

const std::unordered_map<std::string, std::vector<UpgraderEntry>>&
get_operator_version_map() {
  return operatorVersionMap;
}

void test_only_add_entry(std::string op_name, UpgraderEntry entry) {
  operatorVersionMap[op_name].push_back(entry);
}

void test_only_remove_entry(std::string op_name) {
  operatorVersionMap.erase(op_name);
}

} // namespace jit
} // namespace torch
