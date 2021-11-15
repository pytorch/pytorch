#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

struct UpgraderEntry {
  int version_bump;
  std::string upgrader_name;
  std::string old_schema;
};

static std::unordered_map<std::string, std::vector<UpgraderEntry>> operator_version_map(
    {{"div.Tensor",
      {{4,
        "div_Tensor_0_3",
        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},
     {"div.Scalar",
      {{4,
        "div_Scalar_0_3",
        "aten::div.Scalar(Tensor self, Scalar other) -> Tensor"}}},
     {"div.out",
      {{4,
        "div_out_0_3",
        "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"}}},
     {"div_.Tensor",
      {{4,
        "div__Tensor_0_3",
        "aten::div_.Tensor(Tensor(a!), Tensor other) -> Tensor(a!)"}}},
     {"div_.Scalar",
      {{4,
        "div__Scalar_0_3",
        "aten::div_.Scalar(Tensor(a!), Tensor other) -> Tensor(a!)"}}},
     {"full.names",
      {{5,
        "full_names_0_4",
        "aten::full.names(int{} size, Scalar fill_value, *, Dimname{}? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"full.out",
      {{5,
        "full_out_0_4",
        "aten::full.out(int{} size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"}}}});

} // namespace jit
} // namespace torch
