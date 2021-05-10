#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Peephole Optimizes List Ops such as len(li) and li[1].
// Currently this is invoked as part of PeepholeOptimize
// return true if graph is modified
TORCH_API bool PeepholeOptimizeListIdioms(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
