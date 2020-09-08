#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Inline function and method calls.
TORCH_API void Inline(Graph& graph);

// Resconstruct scope from inlined call stack
TORCH_API void ReconstructScopeFromInlinedCallStack(torch::jit::Graph& g);

} // namespace jit
} // namespace torch
