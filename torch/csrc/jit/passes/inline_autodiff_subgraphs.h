#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void InlineAutodiffSubgraphs(
    std::shared_ptr<Graph>& graph,
    size_t threshold = 5);

}
} // namespace torch
