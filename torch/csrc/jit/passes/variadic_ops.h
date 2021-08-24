#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Replaces the `aten::cat` ops in the given graph with variadic cat ops.
// Returns true if the graph is modified.
TORCH_API bool UseVariadicCat(const std::shared_ptr<Graph>& graph);

TORCH_API bool RemoveListMutationAndUseVariadicCat(
    const std::shared_ptr<Graph>& graph);

// Replaces the `aten::stack` ops in the given graph with variadic cat ops.
// Returns true if the graph is modified.
TORCH_API bool UseVariadicStack(const std::shared_ptr<Graph>& graph);

TORCH_API bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
