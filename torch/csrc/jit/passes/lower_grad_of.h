#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This pass removes 'grad_of' nodes, replacing them with conditionals of
// the form:
// if any_defined(inputs):
//  outputs = <original_computation>
// else:
//  outputs = undefineds
TORCH_API void LowerGradOf(Graph& g);

} // namespace jit
} // namespace torch
