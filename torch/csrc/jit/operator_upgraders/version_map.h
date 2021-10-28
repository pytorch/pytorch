#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace torch {
namespace jit {

struct UpgraderEntry {
    int version_bump;
    std::string upgrader_name;
    std::string old_schema;
};

static std::unordered_map<std::string, std::vector<UpgraderEntry>> operator_version_map({
    {"aten::div.Tensor", {{4, "div_Tensor_0_3", "foo.bar()"}}},
    {"aten::div.Scalar", {{4, "div_Scalar_0_3", "foo.bar()"}}},
    {"aten::div.out", {{4, "div_out_0_3", "foo.bar()"}}},
});

} // namespace jit
} // namespace torch
