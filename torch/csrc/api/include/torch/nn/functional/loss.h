#pragma once

#include <ATen/ExpandUtils.h>
#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    const L1LossOptions& options = {}) {
  return torch::l1_loss(
    input,
    target,
    enumtype::reduction_get_enum(options.reduction()));
}

inline Tensor kl_div(
    const Tensor& input,
    const Tensor& target,
    const KLDivLossOptions& options = {}) {
  torch::Reduction::Reduction reduction_enum;

  if (c10::get_if<enumtype::kMean>(&options.reduction())) {
    TORCH_WARN("reduction: 'mean' divides the total loss by both the batch size and the support size."
               "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
               "'mean' will be changed to behave the same as 'batchmean' in the next major release.");
  }

  // special case for batchmean
  if (c10::get_if<enumtype::kBatchMean>(&options.reduction())) {
    reduction_enum = torch::Reduction::Sum;
  } else {
    reduction_enum = enumtype::reduction_get_enum(options.reduction());
  }

  auto reduced = torch::kl_div(input, target, reduction_enum);

  if (c10::get_if<enumtype::kBatchMean>(&options.reduction()) && input.dim() != 0) {
    reduced = reduced / input.sizes()[0];
  }

  return reduced;
}

inline Tensor mse_loss(
    const Tensor& input,
    const Tensor& target,
    const MSELossOptions& options = {}) {
  if (!(target.sizes() == input.sizes())) {
    TORCH_WARN("Using a target size (", target.sizes(),
               ") that is different to the input size (", input.sizes(), "). ",
               "This will likely lead to incorrect results due to broadcasting. ",
               "Please ensure they have the same size.");
  }
  torch::Tensor ret;
  if (target.requires_grad()) {
    ret = torch::pow(input - target, 2);
    if (!c10::get_if<enumtype::kNone>(&options.reduction())) {
      ret = (c10::get_if<enumtype::kMean>(&options.reduction())) ? torch::mean(ret) : torch::sum(ret);
    }
  } else {
    std::vector<torch::Tensor> broadcast_tensors = torch::broadcast_tensors({input, target});
    auto expanded_input = broadcast_tensors[0];
    auto expanded_target = broadcast_tensors[1];
    ret = torch::mse_loss(
      expanded_input,
      expanded_target,
      enumtype::reduction_get_enum(options.reduction()));
  }
  return ret;
}

inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const BCELossOptions& options = {}) {
  auto reduction_enum = enumtype::reduction_get_enum(options.reduction());

  if (target.sizes() != input.sizes()) {
    TORCH_WARN("Using a target size (", target.sizes(), ") ",
               "that is different to the input size (", input.sizes(), ") is deprecated. ",
               "Please ensure they have the same size.");
  }
  if (input.numel() != target.numel()) {
    TORCH_CHECK(
      false,
      "Target and input must have the same number of elements. target nelement (", target.numel(), ") "
      "!= input nelement (", input.numel(), ")");
  }

  auto weight = options.weight();
  if (weight.defined()) {
    auto new_size = at::infer_size(target.sizes(), weight.sizes());
    weight = weight.expand(new_size);
  }

  return torch::binary_cross_entropy(input, target, weight, reduction_enum);
}

inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    const HingeEmbeddingLossOptions& options = {}) {
  return torch::hinge_embedding_loss(
      input,
      target,
      options.margin(),
      enumtype::reduction_get_enum(options.reduction()));
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
    enumtype::reduction_get_enum(options.reduction())
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
    enumtype::reduction_get_enum(options.reduction()));
}

inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiLabelMarginLossOptions& options = {}) {
  return torch::multilabel_margin_loss(
    input,
    target,
    enumtype::reduction_get_enum(options.reduction()));
}

inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const SoftMarginLossOptions& options = {}) {
  return torch::soft_margin_loss(
    input,
    target,
    enumtype::reduction_get_enum(options.reduction()));
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
      false,
      enumtype::get_enum_name(options.reduction()),
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
      enumtype::reduction_get_enum(options.reduction()));
}

} // namespace functional
} // namespace nn
} // namespace torch
