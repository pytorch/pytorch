#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API bool canRunWithAutograd(Node* node);

TORCH_API void InlineAutodiffSubgraphs(
    std::shared_ptr<Graph>& graph,
    size_t threshold = 5);

} // namespace torch::jit
