#include <torch/nn/options/batchnorm.h>

namespace torch::nn {

BatchNormOptions::BatchNormOptions(int64_t num_features)
    : num_features_(num_features) {}

} // namespace torch::nn
