#pragma once

#include <c10/macros/Export.h>

#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Insert sync at end of for-loops to prevent write-after-read race condition.
//!
//! WAR race condition occurs when the next iteration of the loop overwrites
//! shared memory value before a previous operation has finished reading it.
std::vector<Expr*> insertWarThreadSynchronization(
    const std::vector<Expr*>& exprs);

//! Insert syncs between writing to shared memory and then reading it.
//! RAW pass is run before indexing, unrolling (loop duplication), memory
//! aliasing, and index (grid/block bcast/reduction)
std::vector<Expr*> insertRawThreadSynchronization(
    const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
