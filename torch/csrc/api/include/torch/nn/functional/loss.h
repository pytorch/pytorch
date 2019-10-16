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
  TORCH_CHECK(options.p() == 1 || options.p() == 2, "only p == 1 and p == 2 supported");
  if (options.weight().defined()) {
    TORCH_CHECK(options.weight().dim() == 1, "weight must be one-dimensional");
  }

  return torch::multi_margin_loss(
    input,
    target,
    options.p(),
    options.margin(),
    options.weight(),
    options.reduction()
  );
}

inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    const CosineEmbeddingLossOptions& options) {
  return torch::cosine_embedding_loss(
      input1, input2, target, options.margin(), options.reduction());
}

inline Tensor _smooth_l1_loss(const Tensor& input, const Tensor& target) {
    auto t = torch::abs(input - target);
    return torch::where(t < 1, 0.5 * torch::pow(t, 2), t - 0.5);
}

inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    const SmoothL1LossOptions& options = {}) {
  if (target.sizes() != input.sizes()) {
    TORCH_WARN("Using a target size (", target.sizes(), ") that is different to the input size (", input.sizes(), "). ",
                  "This will likely lead to incorrect results due to broadcasting. ",
                  "Please ensure they have the same size.");
  }

  Tensor ret;

  if (target.requires_grad()) {
    ret = _smooth_l1_loss(input, target);
    if (options.reduction() != Reduction::None) {
      ret = options.reduction() == Reduction::Mean ? torch::mean(ret) : torch::sum(ret);
    }
  } else {
    std::vector<Tensor> expanded_tensors = torch::broadcast_tensors({input, target});
    ret = torch::smooth_l1_loss(expanded_tensors[0], expanded_tensors[1], options.reduction());
  }
  return ret;
}
  
inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiLabelMarginLossOptions& options = {}) {
  return torch::multilabel_margin_loss(input, target, options.reduction());
}

inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const SoftMarginLossOptions& options = {}) {
  return torch::soft_margin_loss(input, target, options.reduction());
}

inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiLabelSoftMarginLossOptions& options = {}) {
  auto loss = -(target * torch::log_sigmoid(input) + (1 - target) * torch::log_sigmoid(-input));
  if (options.weight().defined()) {
    loss = loss * options.weight();
  }

  loss = loss.sum(1) / input.size(1); // only return N loss values

  Tensor ret;

  if (options.reduction() == torch::Reduction::None) {
      ret = loss;
  } else if (options.reduction() == torch::Reduction::Mean) {
      ret = loss.mean();
  } else if (options.reduction() == torch::Reduction::Sum) {
      ret = loss.sum();
  } else {
      ret = input;
      TORCH_INTERNAL_ASSERT(true, options.reduction(), " is not valid");
  }
  return ret;
}

inline Tensor triplet_margin_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    const TripletMarginLossOptions& options = {}) {
  return torch::triplet_margin_loss(
      anchor,
      positive,
      negative,
      options.margin(),
      options.p(),
      options.eps(),
      options.swap(),
      options.reduction());
}

} // namespace functional
} // namespace nn
} // namespace torch
