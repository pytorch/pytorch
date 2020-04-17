#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct MutationRemover;

TORCH_API void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph);

TORCH_API void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph);

// Replaces list mutation & in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics.
TORCH_API void RemoveMutation(
    const std::shared_ptr<Graph>& graph,
    bool remove_list_mutation = true,
    bool remove_tensor_mutation = true);

} // namespace jit
} // namespace torch
