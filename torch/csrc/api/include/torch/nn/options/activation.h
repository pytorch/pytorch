#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for ELU functional and module.
struct ELUOptions {
  ELUOptions() {}

  /// The alpha value for the ELU formulation. Default: 1.0
  TORCH_ARG(double, alpha) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for Hardshrink functional and module.
struct TORCH_API HardshrinkOptions {
  /* implicit */ HardshrinkOptions(double lambda = 0.5);

  /// the lambda value for the Hardshrink formulation. Default: 0.5
  TORCH_ARG(double, lambda);
};

// ============================================================================

/// Options for Hardtanh functional and module.
struct HardtanhOptions {
  HardtanhOptions() {}

  /// minimum value of the linear region range. Default: -1
  TORCH_ARG(double, min_val) = -1.0;

  /// maximum value of the linear region range. Default: 1
  TORCH_ARG(double, max_val) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for LeakyReLU functional and module.
struct LeakyReLUOptions {
  LeakyReLUOptions() {}

  /// Controls the angle of the negative slope. Default: 1e-2
  TORCH_ARG(double, negative_slope) = 1e-2;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for the Softmax functional and module.
struct TORCH_API SoftmaxOptions {
  SoftmaxOptions(int dim = -1, torch::Dtype dtype = torch::Dtype::Undefined);

  // Dimension along which Softmax will be computed.
  TORCH_ARG(int, dim);

  // The desired data type of returned tensor.
  // If specified, the input tensor is casted to dtype before the operation
  // is performed. This is useful for preventing data type overflows.
  TORCH_ARG(torch::Dtype, dtype);
};

} // namespace nn
} // namespace torch
