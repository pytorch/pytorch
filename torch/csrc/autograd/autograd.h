#pragma once

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

namespace torch {
namespace autograd {
  /// Computes the gradient of current tensor w.r.t. graph leaves.
  TORCH_API void backward(
      at::TensorList tensors,
      at::TensorList grad_tensors={},
      c10::optional<bool> keep_graph=c10::nullopt,
      bool create_graph=false);

  // Computes the gradient of current tensor w.r.t. inputs.
  TORCH_API variable_list grad(
      at::TensorList outputs,
      at::TensorList inputs,
      at::TensorList grad_outputs={},
      c10::optional<bool> keep_graph=c10::nullopt,
      bool create_graph=false,
      bool allow_unused=false);

}} // namespace torch::autograd
