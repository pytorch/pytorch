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

inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    const CosineEmbeddingLossOptions& options) {
  return torch::cosine_embedding_loss(
      input1, input2, target, options.margin(), options.reduction());
}

} // namespace functional
} // namespace nn
} // namespace torch
