#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace torch {
namespace jit {

using UpgraderDB = std::vector<std::pair<int, std::string>>;

static std::unordered_map<std::string, UpgraderDB> operator_version_map({
    {"aten::div.Tensor", {{4, "div_Tensor_0_3"}}},
    {"aten::div.Scalar", {{4, "div_Scalar_0_3"}}},
    {"aten::div.out", {{4, "div_out_0_3"}}},
});

} // namespace jit
} // namespace torch
