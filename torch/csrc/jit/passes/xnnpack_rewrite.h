#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
TORCH_API void insertXNNPACKOps(std::shared_ptr<Graph>& graph);
TORCH_API void insertXNNPACKOps(script::Module& module);
TORCH_API void FoldXNNPACKPrePackingOps(script::Module& module);
} // namespace jit
} // namespace torch
