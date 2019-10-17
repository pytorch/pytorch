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
      torch::enumtype::_convert_reduction_variant_type_to_enum(options.reduction()));
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
    torch::enumtype::_convert_reduction_variant_type_to_enum(options.reduction())
  );
}

inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    const CosineEmbeddingLossOptions& options) {
  return torch::cosine_embedding_loss(
    input1,
    input2,
    target,
    options.margin(),
    torch::enumtype::_convert_reduction_variant_type_to_enum(options.reduction()));
}

inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiLabelMarginLossOptions& options = {}) {
  return torch::multilabel_margin_loss(
    input,
    target,
    torch::enumtype::_convert_reduction_variant_type_to_enum(options.reduction()));
}

inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const SoftMarginLossOptions& options = {}) {
  return torch::soft_margin_loss(
    input,
    target,
    torch::enumtype::_convert_reduction_variant_type_to_enum(options.reduction()));
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

  if (c10::get_if<enumtype::kNone>(&options.reduction())) {
    ret = loss;
  } else if (c10::get_if<enumtype::kMean>(&options.reduction())) {
    ret = loss.mean();
  } else if (c10::get_if<enumtype::kSum>(&options.reduction())) {
    ret = loss.sum();
  } else {
    ret = input;
    TORCH_INTERNAL_ASSERT(
      true,
      c10::visit(torch::enumtype::enum_name{}, options.reduction()),
      " is not valid");
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
      torch::enumtype::_convert_reduction_variant_type_to_enum(options.reduction()));
}

} // namespace functional
} // namespace nn
} // namespace torch
