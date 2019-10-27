#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

struct TORCH_API InstanceNormOptions {
  /* implicit */ InstanceNormOptions(int64_t num_features);
  TORCH_ARG(int64_t, num_features);

  TORCH_ARG(double, eps) = 1e-5;

  TORCH_ARG(double, momentum) = 0.1;

  TORCH_ARG(bool, affine) = false;

  TORCH_ARG(bool, track_running_stats) = false;
};
} // namespace nn
} // namespace torch
