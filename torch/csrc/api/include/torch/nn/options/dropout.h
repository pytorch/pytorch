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

/// Options for `AlphaDropout` module.
using AlphaDropoutOptions = DropoutOptions;

/// Options for `FeatureAlphaDropout` module.
using FeatureAlphaDropoutOptions = DropoutOptions;

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

/// Options for `AlphaDropout` functional.
struct TORCH_API AlphaDropoutFuncOptions {
  TORCH_ARG(double, p) = 0.5;

  TORCH_ARG(bool, training) = false;

  TORCH_ARG(bool, inplace) = false;
};

/// Options for `FeatureAlphaDropout` functional.
struct TORCH_API FeatureAlphaDropoutFuncOptions {
  TORCH_ARG(double, p) = 0.5;

  TORCH_ARG(bool, training) = false;

  TORCH_ARG(bool, inplace) = false;
};

} // namespace functional

} // namespace nn
} // namespace torch
