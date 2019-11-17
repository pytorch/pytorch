#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>


namespace torch {
namespace nn {

/// Options for `Dropout` module.
struct TORCH_API DropoutOptions {
  /* implicit */ DropoutOptions(double p = 0.5);

  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

/// Options for `Dropout2d` module.
using Dropout2dOptions = DropoutOptions;

/// Options for `Dropout3d` module.
using Dropout3dOptions = DropoutOptions;

/// Options for `FeatureDropout` module.
using FeatureDropoutOptions = DropoutOptions;

namespace functional {

/// Options for `Dropout` functional.
struct TORCH_API DropoutFuncOptions {
  /// The probability of an element to be zeroed. Default: 0.5
  TORCH_ARG(double, p) = 0.5;

  TORCH_ARG(bool, training) = true;

  /// can optionally do the operation in-place. Default: False
  TORCH_ARG(bool, inplace) = false;
};

using Dropout2dFuncOptions = DropoutFuncOptions;

using Dropout3dFuncOptions = DropoutFuncOptions;

} // namespace functional

} // namespace nn
} // namespace torch
