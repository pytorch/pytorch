#pragma once

#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor hinge_embedding_loss(
    const Tensor& x1,
    const Tensor& x2,
    const HingeEmbeddingLossOptions& options) {
  return torch::hinge_embedding_loss(
      x1,
      x2,
      options.margin(),
      options.reduction());
}

inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiMarginLossOptions& options = {}) {
  return torch::multi_margin_loss(
    input,
    target,
    options.p(),
    options.margin(),
    options.weight().value(),
    options.reduction()
  );
}

} // namespace functional
} // namespace nn
} // namespace torch
