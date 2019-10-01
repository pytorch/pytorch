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
