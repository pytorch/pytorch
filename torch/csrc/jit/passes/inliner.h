#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Inline function and method calls.
TORCH_API void Inline(Graph& graph);

} // namespace jit
} // namespace torch
