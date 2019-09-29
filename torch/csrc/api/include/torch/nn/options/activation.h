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

} // namespace nn
} // namespace torch
