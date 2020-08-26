#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Because differentiable graphs detach the gradients of input Tensors,
// creating and inlining differentiable graphs changes the requires_grad
// property of tensors in the graph. This pass updates prim::profiles
// requires_grad to keep profiled properties up to date.
TORCH_API void UpdateDifferentiableGraphRequiresGrad(
    std::shared_ptr<Graph>& diff_forward_graph,
    c10::optional<bool> new_requires_grad);

} // namespace jit
} // namespace torch
