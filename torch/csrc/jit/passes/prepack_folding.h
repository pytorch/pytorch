#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FoldPrePackingOps(script::Module& m,
    const std::unordered_set<std::string>& foldable_prepacking_ops);

} // namespace jit
} // namespace torch
