#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

struct UpgraderEntry {
  size_t version_bump;
  std::string upgrader_name;
  std::string old_schema;
};

static std::unordered_map<std::string, std::vector<UpgraderEntry>> operator_version_map(
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
        "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"}}}});

} // namespace jit
} // namespace torch
