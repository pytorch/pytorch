#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Insert magic zero definition at the begining of the kernel. Insert magic
//! zero update after every (outer most) loop nest with a compile time extent.
//!
//! This will make sure nvrtc does not aggressively save predicate and indices.
std::vector<kir::Expr*> insertMagicZero(const std::vector<kir::Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
