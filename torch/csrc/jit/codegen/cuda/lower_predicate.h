#pragma once
#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Update predicates with valid bool conditionals
//!
std::vector<Expr*> generateConditionalFromPredicate(
    const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
