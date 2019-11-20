#pragma once

#include <ATen/ExpandUtils.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    L1LossFuncOptions::reduction_t reduction) {
  return torch::l1_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    const L1LossFuncOptions& options = {}) {
  return detail::l1_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor kl_div(
    const Tensor& input,
    const Tensor& target,
    KLDivFuncOptions::reduction_t reduction) {
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
    const KLDivFuncOptions& options = {}) {
  return detail::kl_div(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor mse_loss(
    const Tensor& input,
    const Tensor& target,
    MSELossFuncOptions::reduction_t reduction) {
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
    const MSELossFuncOptions& options = {}) {
  return detail::mse_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    BinaryCrossEntropyFuncOptions::reduction_t reduction) {
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

  auto weight_ = weight;
  if (weight_.defined()) {
    auto new_size = at::infer_size(target.sizes(), weight_.sizes());
    weight_ = weight_.expand(new_size);
  }

  return torch::binary_cross_entropy(input, target, weight_, reduction_enum);
}
} // namespace detail

inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const BinaryCrossEntropyFuncOptions& options = {}) {
  return detail::binary_cross_entropy(input, target, options.weight(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    double margin,
    HingeEmbeddingLossFuncOptions::reduction_t reduction) {
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
    const HingeEmbeddingLossFuncOptions& options = {}) {
  return detail::hinge_embedding_loss(input, target, options.margin(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t p,
    double margin,
    const Tensor& weight,
    MultiMarginLossFuncOptions::reduction_t reduction) {
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
    const MultiMarginLossFuncOptions& options = {}) {
  return detail::multi_margin_loss(input, target, options.p(), options.margin(), options.weight(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    double margin,
    CosineEmbeddingLossFuncOptions::reduction_t reduction) {
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
    const CosineEmbeddingLossFuncOptions& options = {}) {
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
    SmoothL1LossFuncOptions::reduction_t reduction) {
  if (target.sizes() != input.sizes()) {
    TORCH_WARN("Using a target size (", target.sizes(), ") that is different to the input size (", input.sizes(), "). ",
                  "This will likely lead to incorrect results due to broadcasting. ",
                  "Please ensure they have the same size.");
  }

  Tensor ret;

  if (target.requires_grad()) {
    ret = _smooth_l1_loss(input, target);
    if (!c10::get_if<enumtype::kNone>(&reduction)) {
      ret = c10::get_if<enumtype::kMean>(&reduction) ? torch::mean(ret) : torch::sum(ret);
    }
  } else {
    std::vector<Tensor> expanded_tensors = torch::broadcast_tensors({input, target});
    ret = torch::smooth_l1_loss(expanded_tensors[0], expanded_tensors[1], enumtype::reduction_get_enum(reduction));
  }
  return ret;
}
} // namespace detail

inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    const SmoothL1LossFuncOptions& options = {}) {
  return detail::smooth_l1_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    MultiLabelMarginLossFuncOptions::reduction_t reduction) {
  return torch::multilabel_margin_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiLabelMarginLossFuncOptions& options = {}) {
  return detail::multilabel_margin_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    SoftMarginLossFuncOptions::reduction_t reduction) {
  return torch::soft_margin_loss(
    input,
    target,
    enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const SoftMarginLossFuncOptions& options = {}) {
  return detail::soft_margin_loss(input, target, options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    MultiLabelSoftMarginLossFuncOptions::reduction_t reduction) {
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
    const MultiLabelSoftMarginLossFuncOptions& options = {}) {
  return detail::multilabel_soft_margin_loss(input, target, options.weight(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor triplet_margin_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    double margin,
    double p,
    double eps,
    bool swap,
    TripletMarginLossFuncOptions::reduction_t reduction) {
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
    const TripletMarginLossFuncOptions& options = {}) {
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
                       int64_t blank,
                       CTCLossFuncOptions::reduction_t reduction,
                       bool zero_infinity) {
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
                       const CTCLossFuncOptions& options = {}) {
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
                               bool log_input,
                               bool full,
                               double eps,
                               PoissonNLLLossFuncOptions::reduction_t reduction) {
  return torch::poisson_nll_loss(
    input, target,
    log_input, full, eps, enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor poisson_nll_loss(const Tensor& input, const Tensor& target,
                               const PoissonNLLLossFuncOptions& options = {}) {
  return detail::poisson_nll_loss(
    input, target,
    options.log_input(), options.full(), options.eps(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor margin_ranking_loss(const Tensor& input1,
                                  const Tensor& input2,
                                  const Tensor& target,
                                  double margin,
                                  MarginRankingLossFuncOptions::reduction_t reduction) {
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
  const Tensor& target, const MarginRankingLossFuncOptions& options = {}) {
  return detail::margin_ranking_loss(input1, input2, target, options.margin(), options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    const NLLLossFuncOptions::reduction_t reduction) {
  if (input.dim() < 2){
    TORCH_CHECK(false, "Expected 2 or more dimensions (got ", input.dim(), ")");
  }

  if (input.sizes()[0] != target.sizes()[0]) {
    TORCH_CHECK(false, "Expected input batch_size (", input.sizes()[0], ") to match target batch_size (", target.sizes()[0], ").");
  }

  torch::Tensor ret;
  torch::Tensor input_ = input;
  torch::Tensor target_ = target;
  if (input_.dim() == 2) {
    ret = torch::nll_loss(
          input_,
          target_,
          weight,
          enumtype::reduction_get_enum(reduction),
          ignore_index);
  } else if (input_.dim() == 4) {
    ret = torch::nll_loss2d(
          input_,
          target_,
          weight,
          enumtype::reduction_get_enum(reduction),
          ignore_index);
  } else {
    // dim == 3 or dim > 4
    auto n = input_.sizes()[0];
    auto c = input_.sizes()[1];
    auto out_size = input_.sizes().slice(2).vec();
    out_size.insert(out_size.begin(), n);
    if (target_.sizes().slice(1) != input_.sizes().slice(2)) {
      TORCH_CHECK(false, "Expected target size ", at::IntArrayRef(out_size), ", got ", target_.sizes());
    }
    input_ = input_.contiguous();
    target_ = target_.contiguous();
    // support empty batches, see #15870
    if (input_.numel() > 0) {
      input_ = input_.view({n, c, 1, -1});
    } else {
      input_ = input_.view({n, c, 0, 0});
    }
    if (target_.numel() > 0) {
      target_ = target_.view({n, 1, -1});
    } else {
      target_ = target_.view({n, 0, 0});
    }
    auto reduction_enum = enumtype::reduction_get_enum(reduction);
    if (!c10::get_if<enumtype::kNone>(&reduction)) {
      ret = torch::nll_loss2d(input_, target_, weight, reduction_enum, ignore_index);
    } else {
      auto out = torch::nll_loss2d(input_, target_, weight, reduction_enum, ignore_index);
      ret = out.view(out_size);
    }
  }
  return ret;
}
} // namespace detail

inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const NLLLossOptions& options = {}) {
  return detail::nll_loss(
    input,
    target,
    options.weight(),
    options.ignore_index(),
    options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    CrossEntropyFuncOptions::reduction_t reduction) {
  NLLLossFuncOptions::reduction_t reduction_;
  if (c10::get_if<enumtype::kNone>(&reduction)) {
    reduction_ = torch::kNone;
  } else if (c10::get_if<enumtype::kMean>(&reduction)) {
    reduction_ = torch::kMean;
  } else if (c10::get_if<enumtype::kSum>(&reduction)) {
    reduction_ = torch::kSum;
  } else {
    TORCH_INTERNAL_ASSERT(
      false,
      enumtype::get_enum_name(reduction),
      " is not valid");
  }
  return torch::nn::functional::detail::nll_loss(
    torch::nn::functional::detail::log_softmax(input, 1, c10::nullopt),
    target,
    weight,
    ignore_index,
    reduction_);
}
} // namespace detail

inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const CrossEntropyFuncOptions& options = {}) {
  return detail::cross_entropy(
      input,
      target,
      options.weight(),
      options.ignore_index(),
      options.reduction());
}

// ============================================================================

namespace detail {
inline Tensor binary_cross_entropy_with_logits(
  const Tensor& input, const Tensor& target, const Tensor& weight,
  BCEWithLogitsLossOptions::reduction_t reduction, const Tensor& pos_weight) {

  TORCH_CHECK(target.sizes() == input.sizes(),
    "Target size (", target.sizes(),
    ") must be the same as input size (",
    input.sizes(), ")"
  );

  return torch::binary_cross_entropy_with_logits(input, target,
    weight, pos_weight, enumtype::reduction_get_enum(reduction));
}
} // namespace detail

inline Tensor binary_cross_entropy_with_logits(
  const Tensor& input, const Tensor& target,
  const BinaryCrossEntropyWithLogitsFuncOptions& options = {}) {
  return detail::binary_cross_entropy_with_logits(input, target,
    options.weight(), options.reduction(), options.pos_weight());
}

} // namespace functional
} // namespace nn
} // namespace torch
