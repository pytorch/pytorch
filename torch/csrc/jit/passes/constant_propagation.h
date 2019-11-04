#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void ConstantPropagation(std::shared_ptr<Graph>& graph);

// Runs the node if its inputs are constants. Callers of this function must
// make their own determination if constant prop is appropriate - for example
// non-deterministic ops or ops with side effects
TORCH_API c10::optional<Stack> runNodeIfInputsAreConstant(const Node* node);

} // namespace jit
} // namespace torch
