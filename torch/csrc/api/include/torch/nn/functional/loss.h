#pragma once

#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor l1_loss(
    const Tensor& self,
    const Tensor& target,
    const L1LossOptions& options) {
  return torch::l1_loss(self, target, options.reduction());
}

inline Tensor kl_div(
    const Tensor& self,
    const Tensor& target,
    const KLDivLossOptions& options) {
  return torch::kl_div(self, target, options.reduction());
}

inline Tensor mse_loss(
    const Tensor& self,
    const Tensor& target,
    const MSELossOptions& options) {
  return torch::mse_loss(self, target, options.reduction());
}

inline Tensor hinge_embedding_loss(
    const Tensor& self,
    const Tensor& target,
    const HingeEmbeddingLossOptions& options) {
  return torch::hinge_embedding_loss(
      self, target, options.margin(), options.reduction());
}

} // namespace functional
} // namespace nn
} // namespace torch
