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
std::vector<Expr*> insertMagicZero(const std::vector<Expr*>& exprs);

//! Check if val is a reference to the magic zero variable
bool isMagicZero(const Val* val);

//! Check if val is protected with magic zero.
//!
//! Specifically, this returns true if val is defined as "x + magic_zero".
bool isProtectedWithMagicZero(const Val* val);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
