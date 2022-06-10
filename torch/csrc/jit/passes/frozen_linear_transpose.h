#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Transposes the weight matrix for frozen linear modules.
// and converts it into a matmul
TORCH_API bool FrozenLinearTranspose(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
