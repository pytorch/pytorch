#pragma once
#include <c10/macros/Export.h>

#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

//! Update predicates with valid bool conditionals
//!
std::vector<Expr*> generateConditionalFromPredicate(
    const std::vector<Expr*>& exprs);

} // namespace nvfuser
