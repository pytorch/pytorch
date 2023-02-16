#pragma once

#include <ir_base_nodes.h>

namespace nvfuser {

std::vector<Expr*> reorderExprsForComputeAt();

} // namespace nvfuser
