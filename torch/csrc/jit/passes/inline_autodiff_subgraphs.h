#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API bool canRunWithAutograd(Node* node);

TORCH_API void InlineAutodiffSubgraphs(
    std::shared_ptr<Graph>& graph,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    size_t threshold = 5);

} // namespace jit
} // namespace torch
