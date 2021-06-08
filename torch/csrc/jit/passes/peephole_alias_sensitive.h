#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Peephole Optimizes alias sensitive peepholes
// Currently this is invoked as part of PeepholeOptimize
// return true if graph is modified
TORCH_API bool PeepholeOptimizeAliasSensitive(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
