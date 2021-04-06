#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Peephole Optimizes List Ops such as len(li) and li[1].
// Currently this is invoked as part of PeepholeOptimize
TORCH_API void PeepholeOptimizeListIdioms(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
