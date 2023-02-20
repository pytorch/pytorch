#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <fusion.h>
#include <ir_all_nodes.h>
#include <lower_trivial_reductions.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Replaces trivial reductions with Unary Set Ops
void trivialReductionReplacement(Fusion*, const TrivialReductionInfo&);

// Transpose, Shift, Gather, and View Ops with Unary Set Ops
std::vector<Expr*> unarySetOpInserter(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
