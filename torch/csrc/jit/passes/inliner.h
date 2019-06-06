#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void Inline(Graph& graph);

} // namespace jit
} // namespace torch
