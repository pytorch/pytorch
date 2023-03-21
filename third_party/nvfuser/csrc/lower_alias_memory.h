#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Reuse Allocation nodes via pointer aliasing
//!
//! First pass finds candidate TensorViews
//! A candidate TensorView is anything in shared memory OR
//! in local memory with a static size larger than register_size_threshold
//!
//! Second pass finds appropriate input Allocate Node
//! among candidate TensorViews
//!
//! Alias Criteria:
//! If input is a candidate TensorView,
//!          input allocation has the same size as output allocation,
//!          thread bindings match,
//!          is not used after this op:
//! then alias output Allocate to input Allocate.
//!
std::vector<Expr*> reuseMemoryAllocations(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
