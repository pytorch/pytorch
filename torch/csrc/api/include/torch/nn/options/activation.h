#pragma once

#include <torch/arg.h>

namespace torch {
namespace nn {

/// Options for ELU functional and module.
struct ELUOptions {
  ELUOptions() {}

  // The alpha value for the ELU formulation. Default: 1.0
  TORCH_ARG(double, alpha) = 1.0;

  // can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

// ============================================================================

/// Options for Hardshrink functional and module.
struct HardshrinkOptions {
  HardshrinkOptions() {}

  // the lambda value for the Hardshrink formulation. Default: 0.5
  TORCH_ARG(double, lambda) = 0.5;
};

// ============================================================================

/// Options for Hardtanh functional and module.
struct HardtanhOptions {
  HardtanhOptions() {}

  // minimum value of the linear region range. Default: -1
  TORCH_ARG(double, min_val) = -1.0;

  // maximum value of the linear region range. Default: 1
  TORCH_ARG(double, max_val) = 1.0;

  // can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

} // namespace nn
} // namespace torch
