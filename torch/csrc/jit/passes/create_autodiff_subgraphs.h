#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <limits>

#include <cstddef>

namespace torch {
namespace jit {

// insert GraphExecutor nodes that group together
// subgraphs that are differentiable by the jit's autodiff passes
// threshold - minimum number of nodes that will appear in a block
// returns all differentiable blocks that have been found
TORCH_API std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold = 2,
    bool strict_requires_grad_check = false,
    size_t max_iterations = std::numeric_limits<size_t>::max());
} // namespace jit
} // namespace torch
