#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Insert sync at end of for-loops to prevent write-after-read race condition.
//! WAR race condition occurs when the next iteration of the loop overwrites
//! shared memory value before a previous operation has finished reading it.
//!
//! WAR Race Check:
//! Track all output shared memory TVs before first sync
//! Track all input shared memory TVs after last sync
//! If the intersection is non-empty, then there is a WAR race condition.
//! Recursively check each nested for-loop
//!
//! Parent-Child For-Loop Recursive Relationship
//! Notation:
//! None - Zero Syncs
//!   1+ - One or more Syncs
//!  End - Sync is last op in for-loop to prevent WAR race condition
//!
//! Default: Track all shared memory inputs and outputs
//!
//! Parent - None
//!  Child - None => Append All Child Outputs to Parent Initial
//!  Child - 1+ => Parent first sync => Inherit Child Initial + Final
//!  Child - End => Parent first sync => Keep Child Initial / Clear Parent Final
//!
//! Parent - 1+
//!  Child - None => Append All Child to Parent Last
//!  Child - 1+ => Child Final to Parent Final / Discard Child Initial
//!  Child - End => Clear Parent Last / Discard Child Initial
//!
//! If Child - End and Parent has zero remaining operations, then
//! Parent inherits Child End.
//!
std::vector<Expr*> insertThreadSynchronization(
    Fusion* fusion,
    const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
