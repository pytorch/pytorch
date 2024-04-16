#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch {
namespace jit {

using PrePackParamFilterFn = std::function<bool(Node*)>;

TORCH_API std::unordered_set<std::string> RegisterPrePackParams(
    Module& m,
    const std::string& method_name,
    const PrePackParamFilterFn& is_packed_param,
    const std::string& attr_prefix);

TORCH_API std::string joinPaths(const std::vector<std::string>& paths);
} // namespace jit
} // namespace torch
