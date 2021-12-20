#pragma once
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

const static std::unordered_map<std::string, std::vector<UpgraderEntry>> kOperatorVersionMap(
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
     {"aten::full.names",
      {{5,
        "full_names_0_4",
        "aten::full.names(int{} size, Scalar fill_value, *, Dimname{}? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"aten::full.out",
      {{5,
        "full_out_0_4",
        "aten::full.out(int{} size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"}}}});

std::unordered_map<std::string, std::vector<UpgraderEntry>>
get_operator_version_map() {
  return kOperatorVersionMap;
}

} // namespace jit
} // namespace torch
