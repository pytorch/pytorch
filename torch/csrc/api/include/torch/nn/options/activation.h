#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/nn/options/common.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for ELU functional and module.
struct TORCH_API ELUOptions {
  /// The `alpha` value for the ELU formulation. Default: 1.0
  TORCH_ARG(double, alpha) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(ELU, ELUFuncOptions)

// ============================================================================

/// Options for SELU functional and module.
struct TORCH_API SELUOptions {
  /* implicit */ SELUOptions(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(SELU, SELUFuncOptions)

// ============================================================================

/// Options for Hardshrink functional and module.
struct TORCH_API HardshrinkOptions {
  /* implicit */ HardshrinkOptions(double lambda = 0.5);

  /// the `lambda` value for the Hardshrink formulation. Default: 0.5
  TORCH_ARG(double, lambda);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Hardshrink, HardshrinkFuncOptions)

// ============================================================================

/// Options for Hardtanh functional and module.
struct TORCH_API HardtanhOptions {
  /// minimum value of the linear region range. Default: -1
  TORCH_ARG(double, min_val) = -1.0;

  /// maximum value of the linear region range. Default: 1
  TORCH_ARG(double, max_val) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Hardtanh, HardtanhFuncOptions)

// ============================================================================

/// Options for LeakyReLU functional and module.
struct TORCH_API LeakyReLUOptions {
  /// Controls the angle of the negative slope. Default: 1e-2
  TORCH_ARG(double, negative_slope) = 1e-2;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(LeakyReLU, LeakyReLUFuncOptions)

// ============================================================================

/// Options for the Softmax functional and module.
struct TORCH_API SoftmaxOptions {
  SoftmaxOptions(int64_t dim);

  /// Dimension along which Softmax will be computed.
  TORCH_ARG(int64_t, dim);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Softmax, SoftmaxFuncOptions)

// ============================================================================

/// Options for the Softmin functional and module.
struct TORCH_API SoftminOptions {
  SoftminOptions(int64_t dim);

  /// Dimension along which Softmin will be computed.
  TORCH_ARG(int64_t, dim);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Softmin, SoftminFuncOptions)

// ============================================================================

/// Options for the LogSoftmax functional and module.
struct TORCH_API LogSoftmaxOptions {
  LogSoftmaxOptions(int64_t dim);

  /// Dimension along which LogSoftmax will be computed.
  TORCH_ARG(int64_t, dim);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(LogSoftmax, LogSoftmaxFuncOptions)

// ============================================================================

/// Options for PReLU functional and module.
struct TORCH_API PReLUOptions {
  /// number of `a` to learn. Although it takes an int as input, there is only
  /// two values are legitimate: 1, or the number of channels at input. Default: 1
  TORCH_ARG(int64_t, num_parameters) = 1;

  /// the initial value of `a`. Default: 0.25
  TORCH_ARG(double, init) = 0.25;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(PReLU, PReLUFuncOptions)

// ============================================================================

/// Options for ReLU functional and module.
struct TORCH_API ReLUOptions {
  /* implicit */ ReLUOptions(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(ReLU, ReLUFuncOptions)

// ============================================================================

/// Options for ReLU6 functional and module.
struct TORCH_API ReLU6Options {
  /* implicit */ ReLU6Options(bool inplace = false);

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(ReLU6, ReLU6FuncOptions)

// ============================================================================

/// Options for RReLU functional and module.
struct TORCH_API RReLUOptions {
  /// lower bound of the uniform distribution. Default: 1/8
  TORCH_ARG(double, lower) = 1.0 / 8.0;

  /// upper bound of the uniform distribution. Default: 1/3
  TORCH_ARG(double, upper) = 1.0 / 3.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(RReLU, RReLUFuncOptions)

// ============================================================================

/// Options for CELU functional and module.
struct TORCH_API CELUOptions {
  /// The `alpha` value for the CELU formulation. Default: 1.0
  TORCH_ARG(double, alpha) = 1.0;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(CELU, CELUFuncOptions)

// ============================================================================

/// Options for Softplus functional and module.
struct TORCH_API SoftplusOptions {
  /// the `beta` value for the Softplus formulation. Default: 1
  TORCH_ARG(double, beta) = 1.0;

  /// values above this revert to a linear function. Default: 20
  TORCH_ARG(double, threshold) = 20.0;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Softplus, SoftplusFuncOptions)

// ============================================================================

/// Options for Softshrink functional and module.
struct TORCH_API SoftshrinkOptions {
  /* implicit */ SoftshrinkOptions(double lambda = 0.5);

  /// the `lambda` value for the Softshrink formulation. Default: 0.5
  TORCH_ARG(double, lambda);
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Softshrink, SoftshrinkFuncOptions)

// ============================================================================

/// Options for Threshold functional and module.
struct ThresholdOptions {
  ThresholdOptions(double threshold, double value)
   : threshold_(threshold), value_(value) {}

  /// The value to threshold at
  TORCH_ARG(double, threshold);

  /// The value to replace with
  TORCH_ARG(double, value);

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Threshold, ThresholdFuncOptions)

// ============================================================================

namespace functional {

/// Options for Gumbel Softmax functional.
struct GumbelSoftmaxFuncOptions {
  /// non-negative scalar temperature
  TORCH_ARG(double, tau) = 1.0;

  /// returned samples will be discretized as one-hot vectors,
  /// but will be differentiated as if it is the soft sample in autograd. Default: False
  TORCH_ARG(bool, hard) = false;

  /// dimension along which softmax will be computed. Default: -1
  TORCH_ARG(int, dim) = -1;
};

} // namespace functional

} // namespace nn
} // namespace torch
