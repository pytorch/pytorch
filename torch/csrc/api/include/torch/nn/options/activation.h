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

/// Options for PReLU functional and module.
struct PReLUOptions {
  PReLUOptions() {}

  /// number of a to learn. Although it takes an int as input, there is only
  /// two values are legitimate: 1, or the number of channels at input. Default: 1
  TORCH_ARG(int64_t, num_parameters) = 1;

  /// the initial value of a. Default: 0.25
  TORCH_ARG(double, init) = 0.25;
};

// ============================================================================

/// Options for ReLU functional and module.
struct TORCH_API ReLUOptions {
  /* implicit */ ReLUOptions(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

// ============================================================================

/// Options for ReLU6 functional and module.
struct TORCH_API ReLU6Options {
  /* implicit */ ReLU6Options(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

// ============================================================================

/// Options for RReLU functional and module.
struct RReLUOptions {
  RReLUOptions() {}

  /// lower bound of the uniform distribution. Default: 1/8
  TORCH_ARG(double, lower) = 1.0 / 8.0;

  /// upper bound of the uniform distribution. Default: 1/3
  TORCH_ARG(double, upper) = 1.0 / 3.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

} // namespace nn
} // namespace torch
