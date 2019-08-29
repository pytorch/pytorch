#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// Inline function and method calls. If `recurse` is true, inline all nested
// calls as well, resulting in a completely flattened graph.
TORCH_API void Inline(Graph& graph, bool recurse = false);

} // namespace jit
} // namespace torch
