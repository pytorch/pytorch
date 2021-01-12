#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<Expr*> reorderExprsForComputeAt(const std::vector<Expr*>& exprs);

}
} // namespace fuser
} // namespace jit
} // namespace torch