#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a L1 loss module.
struct TORCH_API L1LossOptions {
  L1LossOptions(Reduction::Reduction reduction = Reduction::Mean)
      : reduction_(reduction) {}

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(Reduction::Reduction, reduction);
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
  BCELossOptions(
      Tensor weight = {},
      Reduction::Reduction reduction = Reduction::Mean)
      : weight_(weight), reduction_(reduction) {}

  /// A manual rescaling weight given to the loss of each batch element.
  TORCH_ARG(Tensor, weight);

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(Reduction::Reduction, reduction);
};

// ============================================================================

/// Options for a Hinge Embedding loss functional and module.
struct TORCH_API HingeEmbeddingLossOptions {
  /// Specifies the threshold for which the distance of a negative sample must
  /// reach in order to incur zero loss. Default: 1
  TORCH_ARG(double, margin) = 1.0;
  /// Specifies the reduction to apply to the output. Default: Mean
  TORCH_ARG(Reduction::Reduction, reduction) = Reduction::Mean;
};

} // namespace nn
} // namespace torch
