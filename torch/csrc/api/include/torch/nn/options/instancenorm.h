#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

using InstanceNormOptions = BatchNormOptions;

using InstanceNorm1dOptions = InstanceNormOptions;
using InstanceNorm2dOptions = InstanceNormOptions;
using InstanceNorm3dOptions = InstanceNormOptions;

namespace functional {

/// Options for the `InstanceNorm` functional.
struct TORCH_API InstanceNormFuncOptions {
  TORCH_ARG(Tensor, running_mean) = Tensor();

  TORCH_ARG(Tensor, running_var) = Tensor();

  TORCH_ARG(Tensor, weight) = Tensor();

  TORCH_ARG(Tensor, bias) = Tensor();

  TORCH_ARG(bool, use_input_stats) = true;

  TORCH_ARG(double, momentum) = 0.1;

  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace nn
} // namespace torch
