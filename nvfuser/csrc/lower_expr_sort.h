#pragma once

#include <ir_base_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<Expr*> reorderExprsForComputeAt();

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
