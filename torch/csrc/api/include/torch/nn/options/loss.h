#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `L1Loss` module.
///
/// Example:
/// ```
/// L1Loss model(L1LossOptions(torch::kNone));
/// ```
struct TORCH_API L1LossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(L1LossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::l1_loss`.
///
/// See the documentation for `torch::nn::L1LossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::l1_loss(input, target, F::L1LossFuncOptions(torch::kNone));
/// ```
using L1LossFuncOptions = L1LossOptions;
} // namespace functional

// ============================================================================

/// Options for the `KLDivLoss` module.
///
/// Example:
/// ```
/// KLDivLoss
/// model(KLDivLossOptions().reduction(torch::kNone).log_target(false));
/// ```
struct TORCH_API KLDivLossOptions {
  typedef std::variant<
      enumtype::kNone,
      enumtype::kBatchMean,
      enumtype::kSum,
      enumtype::kMean>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG4(
      KLDivLossOptions,
      reduction,
      kNone,
      kBatchMean,
      kSum,
      kMean)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;

  /// Specifies whether `target` is accepted in the log space. Default: False
  TORCH_ARG(bool, log_target) = false;
};

namespace functional {
/// Options for `torch::nn::functional::kl_div`.
///
/// See the documentation for `torch::nn::KLDivLossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::kl_div(input, target,
/// F::KLDivFuncOptions().reduction(torch::kNone).log_target(false));
/// ```
using KLDivFuncOptions = KLDivLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MSELoss` module.
///
/// Example:
/// ```
/// MSELoss model(MSELossOptions(torch::kNone));
/// ```
struct TORCH_API MSELossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(MSELossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::mse_loss`.
///
/// See the documentation for `torch::nn::MSELossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::mse_loss(input, target, F::MSELossFuncOptions(torch::kNone));
/// ```
using MSELossFuncOptions = MSELossOptions;
} // namespace functional

// ============================================================================

/// Options for the `BCELoss` module.
///
/// Example:
/// ```
/// BCELoss model(BCELossOptions().reduction(torch::kNone).weight(weight));
/// ```
struct TORCH_API BCELossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// A manual rescaling weight given to the loss of each batch element.
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::binary_cross_entropy`.
///
/// See the documentation for `torch::nn::BCELossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy(input, target,
/// F::BinaryCrossEntropyFuncOptions().weight(weight));
/// ```
using BinaryCrossEntropyFuncOptions = BCELossOptions;
} // namespace functional

// ============================================================================

/// Options for the `HingeEmbeddingLoss` module.
///
/// Example:
/// ```
/// HingeEmbeddingLoss
/// model(HingeEmbeddingLossOptions().margin(4).reduction(torch::kNone));
/// ```
struct TORCH_API HingeEmbeddingLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Default: 1
  TORCH_ARG(double, margin) = 1.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::hinge_embedding_loss`.
///
/// See the documentation for `torch::nn::HingeEmbeddingLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hinge_embedding_loss(input, target,
/// F::HingeEmbeddingLossFuncOptions().margin(2));
/// ```
using HingeEmbeddingLossFuncOptions = HingeEmbeddingLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MultiMarginLoss` module.
///
/// Example:
/// ```
/// MultiMarginLoss model(MultiMarginLossOptions().margin(2).weight(weight));
/// ```
struct TORCH_API MultiMarginLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Has a default value of :math:`1`. :math:`1` and :math:`2`
  /// are the only supported values.
  TORCH_ARG(int64_t, p) = 1;
  /// Has a default value of :math:`1`.
  TORCH_ARG(double, margin) = 1.0;
  /// A manual rescaling weight given to each
  /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
  /// treated as if having all ones.
  TORCH_ARG(Tensor, weight) = Tensor();
  /// Specifies the reduction to apply to the output:
  /// ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
  /// applied,
  /// ``'mean'``: the sum of the output will be divided by the number of
  /// elements in the output, ``'sum'``: the output will be summed. Default:
  /// ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::multi_margin_loss`.
///
/// See the documentation for `torch::nn::MultiMarginLossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multi_margin_loss(input, target,
/// F::MultiMarginLossFuncOptions().margin(2).weight(weight));
/// ```
using MultiMarginLossFuncOptions = MultiMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `CosineEmbeddingLoss` module.
///
/// Example:
/// ```
/// CosineEmbeddingLoss model(CosineEmbeddingLossOptions().margin(0.5));
/// ```
struct TORCH_API CosineEmbeddingLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Should be a number from -1 to 1, 0
  /// to 0.5 is suggested. Default: 0.0
  TORCH_ARG(double, margin) = 0.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::cosine_embedding_loss`.
///
/// See the documentation for `torch::nn::CosineEmbeddingLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_embedding_loss(input1, input2, target,
/// F::CosineEmbeddingLossFuncOptions().margin(0.5));
/// ```
using CosineEmbeddingLossFuncOptions = CosineEmbeddingLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MultiLabelMarginLoss` module.
///
/// Example:
/// ```
/// MultiLabelMarginLoss model(MultiLabelMarginLossOptions(torch::kNone));
/// ```
struct TORCH_API MultiLabelMarginLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      MultiLabelMarginLossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::multilabel_margin_loss`.
///
/// See the documentation for `torch::nn::MultiLabelMarginLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_margin_loss(input, target,
/// F::MultilabelMarginLossFuncOptions(torch::kNone));
/// ```
using MultilabelMarginLossFuncOptions = MultiLabelMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `SoftMarginLoss` module.
///
/// Example:
/// ```
/// SoftMarginLoss model(SoftMarginLossOptions(torch::kNone));
/// ```
struct TORCH_API SoftMarginLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      SoftMarginLossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::soft_margin_loss`.
///
/// See the documentation for `torch::nn::SoftMarginLossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::soft_margin_loss(input, target,
/// F::SoftMarginLossFuncOptions(torch::kNone));
/// ```
using SoftMarginLossFuncOptions = SoftMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MultiLabelSoftMarginLoss` module.
///
/// Example:
/// ```
/// MultiLabelSoftMarginLoss
/// model(MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight));
/// ```
struct TORCH_API MultiLabelSoftMarginLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// A manual rescaling weight given to each
  /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
  /// treated as if having all ones.
  TORCH_ARG(Tensor, weight) = Tensor();

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::multilabel_soft_margin_loss`.
///
/// See the documentation for `torch::nn::MultiLabelSoftMarginLossOptions` class
/// to learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::multilabel_soft_margin_loss(input, target,
/// F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone).weight(weight));
/// ```
using MultilabelSoftMarginLossFuncOptions = MultiLabelSoftMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `TripletMarginLoss` module.
///
/// Example:
/// ```
/// TripletMarginLoss
/// model(TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false));
/// ```
struct TORCH_API TripletMarginLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Default: 1
  TORCH_ARG(double, margin) = 1.0;
  /// Specifies the norm degree for pairwise distance. Default: 2
  TORCH_ARG(double, p) = 2.0;
  TORCH_ARG(double, eps) = 1e-6;
  /// The distance swap is described in detail in the paper Learning shallow
  /// convolutional feature descriptors with triplet losses by V. Balntas,
  /// E. Riba et al. Default: False
  TORCH_ARG(bool, swap) = false;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::triplet_margin_loss`.
///
/// See the documentation for `torch::nn::TripletMarginLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_loss(anchor, positive, negative,
/// F::TripletMarginLossFuncOptions().margin(1.0));
/// ```
using TripletMarginLossFuncOptions = TripletMarginLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `TripletMarginWithDistanceLoss` module.
///
/// Example:
/// ```
/// TripletMarginWithDistanceLoss
/// model(TripletMarginWithDistanceLossOptions().margin(3).swap(false));
/// ```
struct TORCH_API TripletMarginWithDistanceLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;
  typedef std::function<Tensor(const Tensor&, const Tensor&)>
      distance_function_t;

  /// Specifies a nonnegative, real-valued function that quantifies the
  /// closeness of two tensors. If not specified, `F::pairwise_distance` will
  /// be used. Default: nullopt
  TORCH_ARG(std::optional<distance_function_t>, distance_function) =
      std::nullopt;
  /// Specifies a nonnegative margin representing the minimum difference
  /// between the positive and negative distances required for the loss to be 0.
  /// Larger margins penalize cases where the negative examples are not distance
  /// enough from the anchors, relative to the positives. Default: 1
  TORCH_ARG(double, margin) = 1.0;
  /// Whether to use the distance swap described in the paper Learning shallow
  /// convolutional feature descriptors with triplet losses by V. Balntas,
  /// E. Riba et al. If True, and if the positive example is closer to the
  /// negative example than the anchor is, swaps the positive example and the
  /// anchor in the loss computation. Default: False
  TORCH_ARG(bool, swap) = false;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::triplet_margin_with_distance_loss`.
///
/// See the documentation for `torch::nn::TripletMarginWithDistanceLossOptions`
/// class to learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::triplet_margin_with_distance_loss(anchor, positive, negative,
/// F::TripletMarginWithDistanceLossFuncOptions().margin(1.0));
/// ```
using TripletMarginWithDistanceLossFuncOptions =
    TripletMarginWithDistanceLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `CTCLoss` module.
///
/// Example:
/// ```
/// CTCLoss
/// model(CTCLossOptions().blank(42).zero_infinity(false).reduction(torch::kSum));
/// ```
struct TORCH_API CTCLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// blank label. Default `0`.
  TORCH_ARG(int64_t, blank) = 0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Whether to zero infinite losses and the associated gradients.
  /// Default: `false`. Infinite losses mainly occur when the inputs are
  /// too short to be aligned to the targets.
  TORCH_ARG(bool, zero_infinity) = false;
};

namespace functional {
/// Options for `torch::nn::functional::ctc_loss`.
///
/// See the documentation for `torch::nn::CTCLossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::ctc_loss(log_probs, targets, input_lengths, target_lengths,
/// F::CTCLossFuncOptions().reduction(torch::kNone));
/// ```
using CTCLossFuncOptions = CTCLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `SmoothL1Loss` module.
///
/// Example:
/// ```
/// SmoothL1Loss model(SmoothL1LossOptions().reduction(torch::kNone).beta(0.5));
/// ```
struct TORCH_API SmoothL1LossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      SmoothL1LossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Specifies the threshold at which to change between L1 and L2 loss.
  /// If beta is not specified, a value of 1.0 will be used.
  /// Default: nullopt
  TORCH_ARG(std::optional<double>, beta) = std::nullopt;
};

namespace functional {
/// Options for `torch::nn::functional::smooth_l1_loss`.
///
/// See the documentation for `torch::nn::SmoothL1LossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::smooth_l1_loss(input, target, F::SmoothL1LossFuncOptions(torch::kNone));
/// ```
using SmoothL1LossFuncOptions = SmoothL1LossOptions;
} // namespace functional

// ============================================================================

/// Options for the `HuberLoss` module.
///
/// Example:
/// ```
/// HuberLoss model(HuberLossOptions().reduction(torch::kNone).delta(0.5));
/// ```
struct TORCH_API HuberLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(
      HuberLossOptions,
      reduction,
      kNone,
      kMean,
      kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Specifies the threshold at which to change between L1 and L2 loss.
  /// Default: 1.0
  TORCH_ARG(double, delta) = 1.0;
};

namespace functional {
/// Options for `torch::nn::functional::huber_loss`.
///
/// See the documentation for `torch::nn::HuberLossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::huber_loss(input, target, F::HuberLossFuncOptions(torch::kNone));
/// ```
using HuberLossFuncOptions = HuberLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `PoissonNLLLoss` module.
///
/// Example:
/// ```
/// PoissonNLLLoss
/// model(PoissonNLLLossOptions().log_input(false).full(true).eps(0.42).reduction(torch::kSum));
/// ```
struct TORCH_API PoissonNLLLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// if true the loss is computed as `exp(input) - target * input`,
  /// if false the loss is `input - target * log(input + eps)`.
  TORCH_ARG(bool, log_input) = true;
  /// whether to compute full loss, i.e. to add the Stirling approximation term
  /// target * log(target) - target + 0.5 * log(2 * pi * target).
  TORCH_ARG(bool, full) = false;
  /// Small value to avoid evaluation of `log(0)` when `log_input = false`.
  /// Default: 1e-8
  TORCH_ARG(double, eps) = 1e-8;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::poisson_nll_loss`.
///
/// See the documentation for `torch::nn::PoissonNLLLossOptions` class to learn
/// what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::poisson_nll_loss(input, target,
/// F::PoissonNLLLossFuncOptions().reduction(torch::kNone));
/// ```
using PoissonNLLLossFuncOptions = PoissonNLLLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `MarginRankingLoss` module.
///
/// Example:
/// ```
/// MarginRankingLoss
/// model(MarginRankingLossOptions().margin(0.5).reduction(torch::kSum));
/// ```
struct TORCH_API MarginRankingLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// Has a default value of `0`.
  TORCH_ARG(double, margin) = 0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::margin_ranking_loss`.
///
/// See the documentation for `torch::nn::MarginRankingLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::margin_ranking_loss(input1, input2, target,
/// F::MarginRankingLossFuncOptions().margin(0.5).reduction(torch::kSum));
/// ```
using MarginRankingLossFuncOptions = MarginRankingLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `NLLLoss` module.
///
/// Example:
/// ```
/// NLLLoss model(NLLLossOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
struct TORCH_API NLLLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// A manual rescaling weight given to each
  /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
  /// treated as if having all ones.
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies a target value that is ignored
  /// and does not contribute to the input gradient.
  TORCH_ARG(int64_t, ignore_index) = -100;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

namespace functional {
/// Options for `torch::nn::functional::nll_loss`.
///
/// See the documentation for `torch::nn::NLLLossOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::nll_loss(input, target,
/// F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
using NLLLossFuncOptions = NLLLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `CrossEntropyLoss` module.
///
/// Example:
/// ```
/// CrossEntropyLoss
/// model(CrossEntropyLossOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
struct TORCH_API CrossEntropyLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;

  /// A manual rescaling weight given to each class. If given, has to be a
  /// Tensor of size C
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies a target value that is ignored
  /// and does not contribute to the input gradient.
  TORCH_ARG(int64_t, ignore_index) = -100;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Specifies the amount of smoothing when computing the loss. Default: 0.0
  TORCH_ARG(double, label_smoothing) = 0.0;
};

namespace functional {
/// Options for `torch::nn::functional::cross_entropy`.
///
/// See the documentation for `torch::nn::CrossEntropyLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cross_entropy(input, target,
/// F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
using CrossEntropyFuncOptions = CrossEntropyLossOptions;
} // namespace functional

// ============================================================================

/// Options for the `BCEWithLogitsLoss` module.
///
/// Example:
/// ```
/// BCEWithLogitsLoss
/// model(BCEWithLogitsLossOptions().reduction(torch::kNone).weight(weight));
/// ```
struct TORCH_API BCEWithLogitsLossOptions {
  typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
      reduction_t;
  /// A manual rescaling weight given to the loss of each batch element.
  /// If given, has to be a Tensor of size `nbatch`.
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// A weight of positive examples.
  /// Must be a vector with length equal to the number of classes.
  TORCH_ARG(Tensor, pos_weight) = {};
};

namespace functional {
/// Options for `torch::nn::functional::binary_cross_entropy_with_logits`.
///
/// See the documentation for `torch::nn::BCEWithLogitsLossOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::binary_cross_entropy_with_logits(input, target,
/// F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight).reduction(torch::kSum));
/// ```
using BinaryCrossEntropyWithLogitsFuncOptions = BCEWithLogitsLossOptions;
} // namespace functional

} // namespace nn
} // namespace torch
