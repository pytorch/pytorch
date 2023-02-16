#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <fusion.h>
#include <ir_all_nodes.h>

#include <vector>

namespace nvfuser {

// Transpose, Shift, Gather, and View Ops with Unary Set Ops
std::vector<Expr*> unarySetOpInserter(const std::vector<Expr*>& exprs);

} // namespace nvfuser
