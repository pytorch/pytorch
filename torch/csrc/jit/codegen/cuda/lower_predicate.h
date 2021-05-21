#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Update predicates with valid bool conditionals
//!
std::vector<kir::Expr*> generateConditionalFromPredicate(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
