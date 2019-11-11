#pragma once

#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/nn/options/common.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a L1 loss module.
struct TORCH_API L1LossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(L1LossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(L1Loss, L1LossFuncOptions)

// ============================================================================

/// Options for a KLDiv loss module.
struct TORCH_API KLDivLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kBatchMean, enumtype::kSum, enumtype::kMean> reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG4(KLDivLossOptions, reduction, kNone, kBatchMean, kSum, kMean)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(KLDivLoss, KLDivLossFuncOptions)

// ============================================================================

/// Options for a MSE loss module.
struct TORCH_API MSELossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(MSELossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(MSELoss, MSELossFuncOptions)

// ============================================================================

/// Options for a BCE loss module.
struct TORCH_API BCELossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// A manual rescaling weight given to the loss of each batch element.
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies the reduction to apply to the output.
  /// ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(BCELoss, BCELossFuncOptions)

// ============================================================================

/// Options for a Hinge Embedding loss functional and module.
struct TORCH_API HingeEmbeddingLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Default: 1
  TORCH_ARG(double, margin) = 1.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(HingeEmbeddingLoss, HingeEmbeddingLossFuncOptions)

// ============================================================================

/// Options for a multi-margin loss functional and module.
struct TORCH_API MultiMarginLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

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
  /// ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
  /// ``'mean'``: the sum of the output will be divided by the number of
  /// elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(MultiMarginLoss, MultiMarginLossFuncOptions)

// ============================================================================

/// Options for a Hinge Embedding loss functional and module.
struct TORCH_API CosineEmbeddingLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Should be a number from -1 to 1, 0
  /// to 0.5 is suggested. Default: 0.0
  TORCH_ARG(double, margin) = 0.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(CosineEmbeddingLoss, CosineEmbeddingLossFuncOptions)

// ============================================================================

/// Options for a multi-label margin loss functional and module.
struct TORCH_API MultiLabelMarginLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(MultiLabelMarginLossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(MultiLabelMarginLoss, MultiLabelMarginLossFuncOptions)

// ============================================================================

/// Options for a soft margin loss functional and module.
struct TORCH_API SoftMarginLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  TORCH_OPTIONS_CTOR_VARIANT_ARG3(SoftMarginLossOptions, reduction, kNone, kMean, kSum)

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(SoftMarginLoss, SoftMarginLossFuncOptions)

// ============================================================================

/// Options for a multi-label soft margin loss functional and module.
struct TORCH_API MultiLabelSoftMarginLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

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

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(MultiLabelSoftMarginLoss, MultiLabelSoftMarginLossFuncOptions)

// ============================================================================

/// Options for a triplet-margin-Loss functional and module.
struct TORCH_API TripletMarginLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

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

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(TripletMarginLoss, TripletMarginLossFuncOptions)

// ============================================================================

/// Options for The Connectionist Temporal Classification loss functional and module.
struct TORCH_API CTCLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// blank label. Default `0`.
  TORCH_ARG(int64_t, blank) = 0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
  /// Whether to zero infinite losses and the associated gradients.
  /// Default: `false`. Infinite losses mainly occur when the inputs are
  /// too short to be aligned to the targets.
  TORCH_ARG(bool, zero_infinity) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(CTCLoss, CTCLossFuncOptions)

// ============================================================================

/// Options for a smooth L1 loss functional and module.
struct TORCH_API SmoothL1LossOptions {
  SmoothL1LossOptions(torch::Reduction::Reduction reduction = torch::Reduction::Mean)
    : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(torch::Reduction::Reduction, reduction);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(SmoothL1Loss, SmoothL1LossFuncOptions)

// ============================================================================

/// Options for PoissonNLLLoss functional and module.
struct TORCH_API PoissonNLLLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

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

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(PoissonNLLLoss, PoissonNLLLossFuncOptions)

// ============================================================================

/// Options for MarginRankingLoss functional and module.
struct TORCH_API MarginRankingLossOptions {
  typedef c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum> reduction_t;

  /// Has a default value of `0`.
  TORCH_ARG(double, margin) = 0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(reduction_t, reduction) = torch::kMean;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(MarginRankingLoss, MarginRankingLossFuncOptions)

} // namespace nn
} // namespace torch
