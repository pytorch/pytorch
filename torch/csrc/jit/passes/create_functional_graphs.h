#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct MutationRemover;

TORCH_API void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph);

TORCH_API void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph);

// Replaces in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics
TORCH_API void RemoveListMutation(const std::shared_ptr<Graph>& graph);

// Removes list mutation with functional equivalents
TORCH_API void RemoveTensorMutation(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
