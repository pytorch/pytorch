#pragma once

#include <ATen/ExpandUtils.h>
#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    L1LossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::l1_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    L1LossFuncOptions options = {}) {
  return detail::l1_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor kl_div(
    const Tensor& input,
    const Tensor& target,
    KLDivLossFuncOptions::reduction_t reduction = torch::kMean) {
  torch::Reduction::Reduction reduction_enum;

  if (c10::get_if<enumtype::kMean>(&reduction)) {
    TORCH_WARN("reduction: 'mean' divides the total loss by both the batch size and the support size."
               "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
               "'mean' will be changed to behave the same as 'batchmean' in the next major release.");
  }

  // special case for batchmean
  if (c10::get_if<enumtype::kBatchMean>(&reduction)) {
    reduction_enum = torch::Reduction::Sum;
  } else {
    reduction_enum = enumtype::reduction_get_enum(reduction);
  }

  auto reduced = torch::kl_div(input, target, reduction_enum);

  if (c10::get_if<enumtype::kBatchMean>(&reduction) && input.dim() != 0) {
    reduced = reduced / input.sizes()[0];
  }

  return reduced;
}
} // namespace detail

inline Tensor kl_div(
    const Tensor& input,
    const Tensor& target,
    KLDivLossFuncOptions options = {}) {
  return detail::kl_div(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor mse_loss(
    const Tensor& input,
    const Tensor& target,
    MSELossFuncOptions::reduction_t reduction = torch::kMean) {
  if (!(target.sizes() == input.sizes())) {
    TORCH_WARN("Using a target size (", target.sizes(),
               ") that is different to the input size (", input.sizes(), "). ",
               "This will likely lead to incorrect results due to broadcasting. ",
               "Please ensure they have the same size.");
  }
  torch::Tensor ret;
  if (target.requires_grad()) {
    ret = torch::pow(input - target, 2);
    if (!c10::get_if<enumtype::kNone>(&reduction)) {
      ret = c10::get_if<enumtype::kMean>(&reduction) ? torch::mean(ret) : torch::sum(ret);
    }
  } else {
    std::vector<torch::Tensor> broadcast_tensors = torch::broadcast_tensors({input, target});
    auto expanded_input = broadcast_tensors[0];
    auto expanded_target = broadcast_tensors[1];
    ret = torch::mse_loss(
      expanded_input,
      expanded_target,
      enumtype::reduction_get_enum(reduction));
  }
  return ret;
}
} // namespace detail

inline Tensor mse_loss(
    const Tensor& input,
    const Tensor& target,
    MSELossFuncOptions options = {}) {
  return detail::mse_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    Tensor weight = {},
    BCELossFuncOptions::reduction_t reduction = torch::kMean) {
  auto reduction_enum = enumtype::reduction_get_enum(reduction);

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

  if (weight.defined()) {
    auto new_size = at::infer_size(target.sizes(), weight.sizes());
    weight = weight.expand(new_size);
  }

  return torch::binary_cross_entropy(input, target, weight, reduction_enum);
}
} // namespace detail

inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    BCELossFuncOptions options = {}) {
  return detail::binary_cross_entropy(input, target, options.weight(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    double margin = 1.0,
    HingeEmbeddingLossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::hinge_embedding_loss(
      input,
      target,
      margin,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    HingeEmbeddingLossFuncOptions options = {}) {
  return detail::hinge_embedding_loss(input, target, options.margin(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t p = 1,
    double margin = 1.0,
    Tensor weight = Tensor(),
    MultiMarginLossFuncOptions::reduction_t reduction = torch::kMean) {
  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");
  if (weight.defined()) {
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  }

  return torch::multi_margin_loss(
    input,
    target,
    p,
    margin,
    weight,
    enumtype::reduction_get_enum(reduction)
  );
}
} // namespace detail

inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    MultiMarginLossFuncOptions options = {}) {
  return detail::multi_margin_loss(input, target, options.p(), options.margin(), options.weight(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    double margin = 0.0,
    CosineEmbeddingLossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::cosine_embedding_loss(
    input1,
    input2,
    target,
    margin,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    CosineEmbeddingLossFuncOptions options = {}) {
  return detail::cosine_embedding_loss(input1, input2, target, options.margin(), options.reduction());
}

// ============================================================================

inline Tensor _smooth_l1_loss(const Tensor& input, const Tensor& target) {
    auto t = torch::abs(input - target);
    return torch::where(t < 1, 0.5 * torch::pow(t, 2), t - 0.5);
}

namespace detail {
inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    torch::Reduction::Reduction reduction = torch::Reduction::Mean) {
  if (target.sizes() != input.sizes()) {
    TORCH_WARN("Using a target size (", target.sizes(), ") that is different to the input size (", input.sizes(), "). ",
                  "This will likely lead to incorrect results due to broadcasting. ",
                  "Please ensure they have the same size.");
  }

  Tensor ret;

  if (target.requires_grad()) {
    ret = _smooth_l1_loss(input, target);
    if (reduction != torch::Reduction::None) {
      ret = reduction == torch::Reduction::Mean ? torch::mean(ret) : torch::sum(ret);
    }
  } else {
    std::vector<Tensor> expanded_tensors = torch::broadcast_tensors({input, target});
    ret = torch::smooth_l1_loss(expanded_tensors[0], expanded_tensors[1], reduction);
  }
  return ret;
}
} // namespace detail

inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    SmoothL1LossFuncOptions options = {}) {
  return detail::smooth_l1_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    MultiLabelMarginLossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::multilabel_margin_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    MultiLabelMarginLossFuncOptions options = {}) {
  return detail::multilabel_margin_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    SoftMarginLossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::soft_margin_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    SoftMarginLossFuncOptions options = {}) {
  return detail::soft_margin_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    Tensor weight = Tensor(),
    MultiLabelSoftMarginLossFuncOptions::reduction_t reduction = torch::kMean) {
  auto loss = -(target * torch::log_sigmoid(input) + (1 - target) * torch::log_sigmoid(-input));
  if (weight.defined()) {
    loss = loss * weight;
  }

  loss = loss.sum(1) / input.size(1); // only return N loss values

  Tensor ret;

  if (c10::get_if<enumtype::kNone>(&reduction)) {
    ret = loss;
  } else if (c10::get_if<enumtype::kMean>(&reduction)) {
    ret = loss.mean();
  } else if (c10::get_if<enumtype::kSum>(&reduction)) {
    ret = loss.sum();
  } else {
    ret = input;
    TORCH_INTERNAL_ASSERT(
      false,
      enumtype::get_enum_name(reduction),
      " is not valid");
  }
  return ret;
}
} // namespace detail

inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    MultiLabelSoftMarginLossFuncOptions options = {}) {
  return detail::multilabel_soft_margin_loss(input, target, options.weight(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor triplet_margin_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    double margin = 1.0,
    double p = 2.0,
    double eps = 1e-6,
    bool swap = false,
    TripletMarginLossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::triplet_margin_loss(
      anchor,
      positive,
      negative,
      margin,
      p,
      eps,
      swap,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor triplet_margin_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    TripletMarginLossFuncOptions options = {}) {
  return detail::triplet_margin_loss(
    anchor,
    positive,
    negative,
    options.margin(),
    options.p(),
    options.eps(),
    options.swap(),
    options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor ctc_loss(const Tensor& log_probs,
                       const Tensor& targets,
                       const Tensor& input_lengths,
                       const Tensor& target_lengths,
                       int64_t blank = 0,
                       CTCLossFuncOptions::reduction_t reduction = torch::kMean,
                       bool zero_infinity = false) {
  return torch::ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank,
    enumtype::reduction_get_enum(reduction),
    zero_infinity);
}
} // namespace detail

inline Tensor ctc_loss(const Tensor& log_probs,
                       const Tensor& targets,
                       const Tensor& input_lengths,
                       const Tensor& target_lengths,
                       CTCLossFuncOptions options = {}) {
  return detail::ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    options.blank(),
    options.reduction(),
    options.zero_infinity());
}

// ============================================================================

namespace detail {
inline Tensor poisson_nll_loss(const Tensor& input,
                               const Tensor& target,
                               bool log_input = true,
                               bool full = false,
                               double eps = 1e-8,
                               PoissonNLLLossFuncOptions::reduction_t reduction = torch::kMean) {
  return torch::poisson_nll_loss(
    input, target,
    log_input, full, eps, enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor poisson_nll_loss(const Tensor& input, const Tensor& target,
                               PoissonNLLLossFuncOptions options = {}) {
  return detail::poisson_nll_loss(
    input, target,
    options.log_input(), options.full(), options.eps(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor margin_ranking_loss(const Tensor& input1,
                                  const Tensor& input2,
                                  const Tensor& target,
                                  double margin = 0,
                                  MarginRankingLossFuncOptions::reduction_t reduction = torch::kMean) {
  TORCH_CHECK(
    input1.dim() != 0 && input2.dim() != 0 && target.dim() != 0,
    "margin_ranking_loss does not support scalars, got sizes: "
    "input1: ", input1.sizes(), ", input2: ", input2.sizes(),
    ", target: ", target.sizes());
  return torch::margin_ranking_loss(input1, input2, target, margin,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor margin_ranking_loss(const Tensor& input1, const Tensor& input2,
  const Tensor& target, MarginRankingLossFuncOptions options = {}) {
  return detail::margin_ranking_loss(input1, input2, target, options.margin(), options.reduction());
}

} // namespace functional
} // namespace nn
} // namespace torch
