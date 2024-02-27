#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Try to replace an op that takes a list input with another op that takes a
// variadic number of arguments.
TORCH_API bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

TORCH_API bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

// Convenient functions for replacing aten::stack/aten::cat with their
// variadic versions.
TORCH_API bool UseVariadicCat(const std::shared_ptr<Graph>& graph);
TORCH_API bool RemoveListMutationAndUseVariadicCat(
    const std::shared_ptr<Graph>& graph);

TORCH_API bool UseVariadicStack(const std::shared_ptr<Graph>& graph);
TORCH_API bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
