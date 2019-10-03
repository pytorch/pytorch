#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `Normalize` module.
struct TORCH_API NormalizeOptions {
  /// The exponent value in the norm formulation. Default: 2
  TORCH_ARG(double, p) = 2;
  /// The dimension to reduce. Default: 1
  TORCH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-12
  TORCH_ARG(double, eps) = 1e-12;
};

} // namespace nn
} // namespace torch
 