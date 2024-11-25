#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// return true if graph is modified
TORCH_API bool PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool disable_shape_peepholes = false);
// return true if graph is modified
TORCH_API bool PeepholeOptimize(
    Block* block,
    bool disable_shape_peepholes = false);
// return true if graph is modified
TORCH_API bool FuseAddMM(const std::shared_ptr<Graph>& graph);

} // namespace torch::jit
