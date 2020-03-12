#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using PrePackingOpsFilterFn = std::function<bool(Node*)>;

void FoldPrePackingOps(
    script::Module& m,
    const PrePackingOpsFilterFn& is_foldable_op,
    const std::string& attr_prefix);

} // namespace jit
} // namespace torch
