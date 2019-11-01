#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `AdaptiveLogSoftmaxWithLoss` module.
struct TORCH_API AdaptiveLogSoftmaxWithLossOptions {
  AdaptiveLogSoftmaxWithLossOptions(int64_t in_features, int64_t n_classes, std::vector<int64_t> cutoffs)
  : in_features_(in_features), n_classes_(n_classes), cutoffs_(cutoffs) {}
 /// Number of features in the input tensor
  TORCH_ARG(int64_t, in_features);

  /// Number of classes in the dataset
  TORCH_ARG(int64_t, n_classes);

  /// Cutoffs used to assign targets to their buckets
  TORCH_ARG(std::vector<int64_t>, cutoffs);

  /// value used as an exponent to compute sizes of the clusters. Default: 4.0
  TORCH_ARG(double, div_value) = 4.;

  /// exponent. Default: 0.75
  TORCH_ARG(bool, head_bias) = false;
};

} // namespace nn
} // namespace torch
