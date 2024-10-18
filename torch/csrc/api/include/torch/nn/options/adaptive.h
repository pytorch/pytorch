#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>

namespace torch::nn {

/// Options for the `AdaptiveLogSoftmaxWithLoss` module.
///
/// Example:
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```
struct TORCH_API AdaptiveLogSoftmaxWithLossOptions {
  /* implicit */ AdaptiveLogSoftmaxWithLossOptions(
      int64_t in_features,
      int64_t n_classes,
      std::vector<int64_t> cutoffs);

  /// Number of features in the input tensor
  TORCH_ARG(int64_t, in_features);

  /// Number of classes in the dataset
  TORCH_ARG(int64_t, n_classes);

  /// Cutoffs used to assign targets to their buckets
  TORCH_ARG(std::vector<int64_t>, cutoffs);

  /// value used as an exponent to compute sizes of the clusters. Default: 4.0
  TORCH_ARG(double, div_value) = 4.;

  /// If ``true``, adds a bias term to the 'head' of
  /// the adaptive softmax. Default: false
  TORCH_ARG(bool, head_bias) = false;
};

} // namespace torch::nn
