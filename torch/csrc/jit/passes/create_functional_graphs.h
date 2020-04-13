#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct MutationRemover;

TORCH_API void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph);

TORCH_API void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph);

// Replaces in-place aten ops with their functional equivalents when it can
// be proven that this does not change graph semantics.
TORCH_API void RemoveMutation(const std::shared_ptr<Graph>& graph);

// Functional Graphs are created with the condition that graph outputs may
// not alias graph outputs. Passes may make optimizations such that this
// is no longer true, such as replacing x + 0 with x.
// This pass takes in a prim::FunctionalGraph and adds copies until outputs do
// not alias inputs.
TORCH_API void EnsureOutputsDontAliasInputs(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
