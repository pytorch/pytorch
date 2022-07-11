#pragma once

#include <ATen/ExpandUtils.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/options/loss.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    L1LossFuncOptions::reduction_t reduction) {
  return torch::l1_loss(input, target, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.l1_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::L1LossFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::l1_loss(input, target, F::L1LossFuncOptions(torch::kNone));
/// ```
inline Tensor l1_loss(
    const Tensor& input,
    const Tensor& target,
    const L1LossFuncOptions& options = {}) {
  return detail::l1_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor kl_div(
    const Tensor& input,
    const Tensor& target,
    KLDivFuncOptions::reduction_t reduction,
    bool log_target = false) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  torch::Reduction::Reduction reduction_enum;

  if (c10::get_if<enumtype::kMean>(&reduction)) {
    TORCH_WARN(
        "reduction: 'mean' divides the total loss by both the batch size and the support size."
        "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
        "'mean' will be changed to behave the same as 'batchmean' in the next major release.");
  }

  // special case for batchmean
  if (c10::get_if<enumtype::kBatchMean>(&reduction)) {
    reduction_enum = torch::Reduction::Sum;
  } else {
    reduction_enum = enumtype::reduction_get_enum(reduction);
  }

  auto reduced = torch::kl_div(input, target, reduction_enum, log_target);

  if (c10::get_if<enumtype::kBatchMean>(&reduction) && input.dim() != 0) {
    reduced = reduced / input.sizes()[0];
  }

  return reduced;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.kl_div
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::KLDivFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::kl_div(input, target,
/// F::KLDivFuncOptions.reduction(torch::kNone).log_target(false));
/// ```
inline Tensor kl_div(
    const Tensor& input,
    const Tensor& target,
    const KLDivFuncOptions& options = {}) {
  return detail::kl_div(
      input, target, options.reduction(), options.log_target());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor mse_loss(
    const Tensor& input,
    const Tensor& target,
    MSELossFuncOptions::reduction_t reduction) {
  if (!(target.sizes() == input.sizes())) {
    TORCH_WARN(
        "Using a target size (",
        target.sizes(),
        ") that is different to the input size (",
        input.sizes(),
        "). ",
        "This will likely lead to incorrect results due to broadcasting. ",
        "Please ensure they have the same size.");
  }
  std::vector<torch::Tensor> broadcast_tensors =
      torch::broadcast_tensors({input, target});
  auto expanded_input = broadcast_tensors[0];
  auto expanded_target = broadcast_tensors[1];
  return torch::mse_loss(
      expanded_input, expanded_target, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.mse_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MSELossFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::mse_loss(input, target, F::MSELossFuncOptions(torch::kNone));
/// ```
inline Tensor mse_loss(
    const Tensor& input,
    const Tensor& target,
    const MSELossFuncOptions& options = {}) {
  return detail::mse_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    BinaryCrossEntropyFuncOptions::reduction_t reduction) {
  auto reduction_enum = enumtype::reduction_get_enum(reduction);

  if (target.sizes() != input.sizes()) {
    TORCH_CHECK(
        false,
        "Using a target size (",
        target.sizes(),
        ") ",
        "that is different to the input size (",
        input.sizes(),
        ") is deprecated. ",
        "Please ensure they have the same size.");
  }

  auto weight_ = weight;
  if (weight_.defined()) {
    auto new_size = at::infer_size(target.sizes(), weight_.sizes());
    weight_ = weight_.expand(new_size);
  }

  return torch::binary_cross_entropy(input, target, weight_, reduction_enum);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.binary_cross_entropy
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::BinaryCrossEntropyFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy(input, target,
/// F::BinaryCrossEntropyFuncOptions().weight(weight));
/// ```
inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const BinaryCrossEntropyFuncOptions& options = {}) {
  return detail::binary_cross_entropy(
      input, target, options.weight(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    double margin,
    HingeEmbeddingLossFuncOptions::reduction_t reduction) {
  return torch::hinge_embedding_loss(
      input, target, margin, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.hinge_embedding_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::HingeEmbeddingLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hinge_embedding_loss(input, target,
/// F::HingeEmbeddingLossFuncOptions().margin(2));
/// ```
inline Tensor hinge_embedding_loss(
    const Tensor& input,
    const Tensor& target,
    const HingeEmbeddingLossFuncOptions& options = {}) {
  return detail::hinge_embedding_loss(
      input, target, options.margin(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.multi_margin_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::MultiMarginLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multi_margin_loss(input, target,
/// F::MultiMarginLossFuncOptions().margin(2).weight(weight));
/// ```
inline Tensor multi_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultiMarginLossFuncOptions& options = {}) {
  return detail::multi_margin_loss(
      input,
      target,
      options.p(),
      options.margin(),
      options.weight(),
      options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    double margin,
    CosineEmbeddingLossFuncOptions::reduction_t reduction) {
  return torch::cosine_embedding_loss(
      input1, input2, target, margin, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.cosine_embedding_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::CosineEmbeddingLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_embedding_loss(input1, input2, target,
/// F::CosineEmbeddingLossFuncOptions().margin(0.5));
/// ```
inline Tensor cosine_embedding_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    const CosineEmbeddingLossFuncOptions& options = {}) {
  return detail::cosine_embedding_loss(
      input1, input2, target, options.margin(), options.reduction());
}

// ============================================================================

inline Tensor _smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    double beta = 1.) {
  auto t = torch::abs(input - target);
  return torch::where(t < beta, 0.5 * torch::pow(t, 2) / beta, t - 0.5 * beta);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    SmoothL1LossFuncOptions::reduction_t reduction,
    double beta = 1.) {
  if (target.sizes() != input.sizes()) {
    TORCH_WARN(
        "Using a target size (",
        target.sizes(),
        ") that is different to the input size (",
        input.sizes(),
        "). ",
        "This will likely lead to incorrect results due to broadcasting. ",
        "Please ensure they have the same size.");
  }

  std::vector<Tensor> expanded_tensors =
      torch::broadcast_tensors({input, target});
  return torch::smooth_l1_loss(
      expanded_tensors[0],
      expanded_tensors[1],
      enumtype::reduction_get_enum(reduction),
      beta);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.smooth_l1_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SmoothL1LossFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::smooth_l1_loss(input, target, F::SmoothL1LossFuncOptions(torch::kNone));
/// ```
inline Tensor smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    const SmoothL1LossFuncOptions& options = {},
    double beta = 1.) {
  return detail::smooth_l1_loss(input, target, options.reduction(), beta);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor huber_loss(
    const Tensor& input,
    const Tensor& target,
    HuberLossFuncOptions::reduction_t reduction,
    double delta = 1.) {
  if (target.sizes() != input.sizes()) {
    TORCH_WARN(
        "Using a target size (",
        target.sizes(),
        ") that is different to the input size (",
        input.sizes(),
        "). ",
        "This will likely lead to incorrect results due to broadcasting. ",
        "Please ensure they have the same size.");
  }

  std::vector<Tensor> expanded_tensors =
      torch::broadcast_tensors({input, target});
  return torch::huber_loss(
      expanded_tensors[0],
      expanded_tensors[1],
      enumtype::reduction_get_enum(reduction),
      delta);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.huber_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::HuberLossFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::huber_loss(input, target,
/// F::HuberLossFuncOptions().reduction(torch::kNone).delta(0.5));
/// ```
inline Tensor huber_loss(
    const Tensor& input,
    const Tensor& target,
    const HuberLossFuncOptions& options = {}) {
  return detail::huber_loss(
      input, target, options.reduction(), options.delta());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    MultilabelMarginLossFuncOptions::reduction_t reduction) {
  return torch::multilabel_margin_loss(
      input, target, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.multilabel_margin_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::MultilabelMarginLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_margin_loss(input, target,
/// F::MultilabelMarginLossFuncOptions(torch::kNone));
/// ```
inline Tensor multilabel_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultilabelMarginLossFuncOptions& options = {}) {
  return detail::multilabel_margin_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    SoftMarginLossFuncOptions::reduction_t reduction) {
  return torch::soft_margin_loss(
      input, target, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.soft_margin_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SoftMarginLossFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::soft_margin_loss(input, target,
/// F::SoftMarginLossFuncOptions(torch::kNone));
/// ```
inline Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const SoftMarginLossFuncOptions& options = {}) {
  return detail::soft_margin_loss(input, target, options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    MultilabelSoftMarginLossFuncOptions::reduction_t reduction) {
  auto loss =
      -(target * torch::log_sigmoid(input) +
        (1 - target) * torch::log_sigmoid(-input));
  if (weight.defined()) {
    loss = loss * weight;
  }

  auto class_dim = input.dim() - 1;
  auto C = input.size(class_dim);
  loss = loss.sum(class_dim) / C; // only return N loss values

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
        false, enumtype::get_enum_name(reduction), " is not valid");
  }
  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.multilabel_soft_margin_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::MultilabelSoftMarginLossFuncOptions` class to learn
/// what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_soft_margin_loss(input, target,
/// F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone).weight(weight));
/// ```
inline Tensor multilabel_soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    const MultilabelSoftMarginLossFuncOptions& options = {}) {
  return detail::multilabel_soft_margin_loss(
      input, target, options.weight(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.triplet_margin_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::TripletMarginLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_loss(anchor, positive, negative,
/// F::TripletMarginLossFuncOptions().margin(1.0));
/// ```
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

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor triplet_margin_with_distance_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    c10::optional<TripletMarginWithDistanceLossFuncOptions::distance_function_t>
        distance_function,
    double margin,
    bool swap,
    TripletMarginWithDistanceLossFuncOptions::reduction_t reduction) {
  Tensor dist_pos, dist_neg;
  if (distance_function.has_value()) {
    auto distance_function_impl = distance_function.value();
    dist_pos = distance_function_impl(anchor, positive);
    dist_neg = distance_function_impl(anchor, negative);
  } else {
    dist_pos = pairwise_distance(anchor, positive);
    dist_neg = pairwise_distance(anchor, negative);
  }

  if (swap) {
    Tensor dist_swap;
    if (distance_function.has_value()) {
      dist_swap = distance_function.value()(positive, negative);
    } else {
      dist_swap = pairwise_distance(positive, negative);
    }
    dist_neg = torch::min(dist_neg, dist_swap);
  }

  auto loss = torch::clamp_min(dist_pos - dist_neg + margin, 0);

  Tensor ret;
  if (c10::get_if<enumtype::kNone>(&reduction)) {
    ret = loss;
  } else if (c10::get_if<enumtype::kMean>(&reduction)) {
    ret = loss.mean();
  } else if (c10::get_if<enumtype::kSum>(&reduction)) {
    ret = loss.sum();
  } else {
    ret = anchor;
    TORCH_INTERNAL_ASSERT(
        false, enumtype::get_enum_name(reduction), " is not valid");
  }
  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.triplet_margin_with_distance_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::TripletMarginWithDistanceLossFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_with_distance_loss(anchor, positive, negative,
/// F::TripletMarginWithDistanceLossFuncOptions().margin(1.0));
/// ```
inline Tensor triplet_margin_with_distance_loss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    const TripletMarginWithDistanceLossFuncOptions& options = {}) {
  return detail::triplet_margin_with_distance_loss(
      anchor,
      positive,
      negative,
      options.distance_function(),
      options.margin(),
      options.swap(),
      options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor ctc_loss(
    const Tensor& log_probs,
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
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.ctc_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::CTCLossFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::ctc_loss(log_probs, targets, input_lengths, target_lengths,
/// F::CTCLossFuncOptions().reduction(torch::kNone));
/// ```
inline Tensor ctc_loss(
    const Tensor& log_probs,
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

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor poisson_nll_loss(
    const Tensor& input,
    const Tensor& target,
    bool log_input,
    bool full,
    double eps,
    PoissonNLLLossFuncOptions::reduction_t reduction) {
  return torch::poisson_nll_loss(
      input,
      target,
      log_input,
      full,
      eps,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.poisson_nll_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::PoissonNLLLossFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::poisson_nll_loss(input, target,
/// F::PoissonNLLLossFuncOptions().reduction(torch::kNone));
/// ```
inline Tensor poisson_nll_loss(
    const Tensor& input,
    const Tensor& target,
    const PoissonNLLLossFuncOptions& options = {}) {
  return detail::poisson_nll_loss(
      input,
      target,
      options.log_input(),
      options.full(),
      options.eps(),
      options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor margin_ranking_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    double margin,
    MarginRankingLossFuncOptions::reduction_t reduction) {
  TORCH_CHECK(
      input1.dim() == input2.dim() && input1.dim() == target.dim(),
      "margin_ranking_loss : All input tensors should have same dimension but got sizes: "
      "input1: ",
      input1.sizes(),
      ", input2: ",
      input2.sizes(),
      ", target: ",
      target.sizes());
  return torch::margin_ranking_loss(
      input1, input2, target, margin, enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.margin_ranking_loss
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::MarginRankingLossFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::margin_ranking_loss(input1, input2, target,
/// F::MarginRankingLossFuncOptions().margin(0.5).reduction(torch::kSum));
/// ```
inline Tensor margin_ranking_loss(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target,
    const MarginRankingLossFuncOptions& options = {}) {
  return detail::margin_ranking_loss(
      input1, input2, target, options.margin(), options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    const NLLLossFuncOptions::reduction_t reduction) {
  if (input.dim() < 2) {
    TORCH_CHECK(false, "Expected 2 or more dimensions (got ", input.dim(), ")");
  }

  if (input.sizes()[0] != target.sizes()[0]) {
    TORCH_CHECK(
        false,
        "Expected input batch_size (",
        input.sizes()[0],
        ") to match target batch_size (",
        target.sizes()[0],
        ").");
  }

  return torch::nll_loss_nd(
      input,
      target,
      weight,
      enumtype::reduction_get_enum(reduction),
      ignore_index);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.nll_loss
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::NLLLossFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::nll_loss(input, target,
/// F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const NLLLossFuncOptions& options = {}) {
  return detail::nll_loss(
      input,
      target,
      options.weight(),
      options.ignore_index(),
      options.reduction());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t ignore_index,
    CrossEntropyFuncOptions::reduction_t reduction,
    double label_smoothing) {
  return torch::cross_entropy_loss(
      input,
      target,
      weight,
      enumtype::reduction_get_enum(reduction),
      ignore_index,
      label_smoothing);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.cross_entropy
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::CrossEntropyFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cross_entropy(input, target,
/// F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const CrossEntropyFuncOptions& options = {}) {
  return detail::cross_entropy(
      input,
      target,
      options.weight(),
      options.ignore_index(),
      options.reduction(),
      options.label_smoothing());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor binary_cross_entropy_with_logits(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    BinaryCrossEntropyWithLogitsFuncOptions::reduction_t reduction,
    const Tensor& pos_weight) {
  TORCH_CHECK(
      target.sizes() == input.sizes(),
      "Target size (",
      target.sizes(),
      ") must be the same as input size (",
      input.sizes(),
      ")");

  return torch::binary_cross_entropy_with_logits(
      input,
      target,
      weight,
      pos_weight,
      enumtype::reduction_get_enum(reduction));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.binary_cross_entropy_with_logits
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy_with_logits(input, target,
/// F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight).reduction(torch::kSum));
/// ```
inline Tensor binary_cross_entropy_with_logits(
    const Tensor& input,
    const Tensor& target,
    const BinaryCrossEntropyWithLogitsFuncOptions& options = {}) {
  return detail::binary_cross_entropy_with_logits(
      input,
      target,
      options.weight(),
      options.reduction(),
      options.pos_weight());
}

} // namespace functional
} // namespace nn
} // namespace torch
