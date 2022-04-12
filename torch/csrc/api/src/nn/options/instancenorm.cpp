#include <torch/nn/options/instancenorm.h>

namespace torch {
namespace nn {

InstanceNormOptions::InstanceNormOptions(int64_t num_features) : num_features_(num_features) {}

} // namespace nn
} // namespace torch
