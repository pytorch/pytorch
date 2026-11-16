#include <torch/nn/options/instancenorm.h>

namespace torch::nn {

InstanceNormOptions::InstanceNormOptions(int64_t num_features)
    : num_features_(num_features) {}

} // namespace torch::nn
