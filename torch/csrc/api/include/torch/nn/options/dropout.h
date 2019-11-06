#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for `Dropout` and `FeatureDropout`.
template <size_t D>
struct DropoutOptions {
  /* implicit */ DropoutOptions(double p = 0.5, bool inplace = false);
  /// The probability with which a particular component of the input is set to
  /// zero.
  /// Changes to this parameter at runtime are effective.
  TORCH_ARG(double, p);

  /// If set to True, will do this operation in-place
  TORCH_ARG(bool, inplace);
};

/// `DropoutOptions` specialized for 1-D Dropout.
using DropoutOptions = DropoutOptions<1>;

/// `DropoutOptions` specialized for 2-D Dropout.
using Dropout2dOptions = DropoutOptions<2>;

/// `DropoutOptions` specialized for 3-D Dropout.
using Dropout3dOptions = DropoutOptions<3>;


} // namespace nn
} // namespace torch
