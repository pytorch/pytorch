#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Peephole Optimizes alias sensitive peepholes
// Currently this is invoked as part of PeepholeOptimize
TORCH_API void PeepholeOptimizeAliasSensitive(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
