#include <torch/nn/options/adaptive.h>

namespace torch {
namespace nn {

AdaptiveLogSoftmaxWithLossOptions::AdaptiveLogSoftmaxWithLossOptions(
    int64_t in_features, int64_t n_classes, std::vector<int64_t> cutoffs)
  : in_features_(in_features), n_classes_(n_classes), cutoffs_(std::move(cutoffs)) {}

} // namespace nn
} // namespace torch
