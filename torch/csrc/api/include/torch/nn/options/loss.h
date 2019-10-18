#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a L1 loss module.
struct TORCH_API L1LossOptions {
  L1LossOptions(torch::Reduction::Reduction reduction = torch::Reduction::Mean)
      : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(torch::Reduction::Reduction, reduction);
};

// ============================================================================

/// Options for a KLDiv loss module.
struct TORCH_API KLDivLossOptions {
  KLDivLossOptions(Reduction::Reduction reduction = Reduction::Mean)
      : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(Reduction::Reduction, reduction);
};

// ============================================================================

/// Options for a MSE loss module.
struct TORCH_API MSELossOptions {
  MSELossOptions(Reduction::Reduction reduction = Reduction::Mean)
      : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(Reduction::Reduction, reduction);
};

// ============================================================================

/// Options for a BCE loss module.
struct TORCH_API BCELossOptions {
  /// A manual rescaling weight given to the loss of each batch element.
  TORCH_ARG(Tensor, weight) = {};
  /// Specifies the reduction to apply to the output.
  TORCH_ARG(Reduction::Reduction, reduction) = Reduction::Mean;
};

// ============================================================================

/// Options for a Hinge Embedding loss functional and module.
struct TORCH_API HingeEmbeddingLossOptions {
  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Default: 1
  TORCH_ARG(double, margin) = 1.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(torch::Reduction::Reduction, reduction) = torch::Reduction::Mean;
};

// ============================================================================

/// Options for a multi-margin loss functional and module.
struct TORCH_API MultiMarginLossOptions {
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
  TORCH_ARG(torch::Reduction::Reduction, reduction) = torch::Reduction::Mean;
};

// ============================================================================

/// Options for a Hinge Embedding loss functional and module.
struct TORCH_API CosineEmbeddingLossOptions {
  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Should be a number from -1 to 1, 0
  /// to 0.5 is suggested. Default: 0.0
  TORCH_ARG(double, margin) = 0.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(torch::Reduction::Reduction, reduction) = torch::Reduction::Mean;
};

// ============================================================================

/// Options for a multi-label margin loss functional and module.
struct TORCH_API MultiLabelMarginLossOptions {
  MultiLabelMarginLossOptions(torch::Reduction::Reduction reduction = torch::Reduction::Mean)
    : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(torch::Reduction::Reduction, reduction);
};

// ============================================================================

/// Options for a soft margin loss functional and module.
struct TORCH_API SoftMarginLossOptions {
  SoftMarginLossOptions(torch::Reduction::Reduction reduction = torch::Reduction::Mean)
    : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(torch::Reduction::Reduction, reduction);
};

// ============================================================================

/// Options for a multi-label soft margin loss functional and module.
struct TORCH_API MultiLabelSoftMarginLossOptions {
  /// A manual rescaling weight given to each
  /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
  /// treated as if having all ones.
  TORCH_ARG(Tensor, weight) = Tensor();

  /// Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
  /// 'none': no reduction will be applied, 'mean': the sum of the output will
  /// be divided by the number of elements in the output, 'sum': the output will
  /// be summed. Default: 'mean'
  TORCH_ARG(torch::Reduction::Reduction, reduction) = torch::Reduction::Mean;
};

// ============================================================================

/// Options for a triplet-margin-Loss functional and module.
struct TORCH_API TripletMarginLossOptions {
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
  TORCH_ARG(torch::Reduction::Reduction, reduction) = torch::Reduction::Mean;
};

} // namespace nn
} // namespace torch
