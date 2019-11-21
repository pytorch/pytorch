#pragma once

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

namespace torch {
namespace autograd {
// Computes the gradient of current tensor w.r.t. graph leaves.
TORCH_API void backward(
    const variable_list& tensors,
    const variable_list& grad_tensors = {},
    c10::optional<bool> retain_graph = c10::nullopt,
    bool create_graph = false);

// Computes the gradient of current tensor w.r.t. inputs.
TORCH_API variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs = {},
    c10::optional<bool> retain_graph = c10::nullopt,
    bool create_graph = false,
    bool allow_unused = false);

} // namespace autograd
} // namespace torch
