#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void Inline(Graph& graph, bool inline_autograd = false);

} // namespace jit
} // namespace torch
