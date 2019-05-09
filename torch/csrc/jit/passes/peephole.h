#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool addmm_fusion_enabled = false);
TORCH_API void PeepholeOptimize(
    Block* block,
    bool addmm_fusion_enabled = false);

} // namespace jit
} // namespace torch
