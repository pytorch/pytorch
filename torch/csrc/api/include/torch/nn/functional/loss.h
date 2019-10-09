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

inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiLabelSoftMarginLossOptions& options) {
  auto loss = -(target * torch::log_sigmoid(input) + (1 - target) * torch::log_sigmoid(-input));
  if (options.weight().defined()) {
    loss = loss * options.weight();
  }

  loss = loss.sum(1) / input.size(1); // only return N loss values

  Tensor ret;

  if (options.reduction() == Reduction::None) {
      ret = loss;
  } else if (options.reduction() == Reduction::Mean) {
      ret = loss.mean();
  } else if (options.reduction() == Reduction::Sum) {
      ret = loss.sum();
  } else {
      ret = input;
      TORCH_INTERNAL_ASSERT(true, options.reduction(), " is not valid");
  }
  return ret;
}

} // namespace functional
} // namespace nn
} // namespace torch
