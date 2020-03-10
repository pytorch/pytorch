#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using PrePackingOpsFilterFn = std::function<bool(Node*)>;

void FoldPrePackingOps(script::Module& m, PrePackingOpsFilterFn is_foldable_op);

} // namespace jit
} // namespace torch
