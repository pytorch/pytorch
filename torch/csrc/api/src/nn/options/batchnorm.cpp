#include <torch/nn/options/batchnorm.h>

namespace torch {
namespace nn {

BatchNormOptions::BatchNormOptions(int64_t features) : features_(features) {}

template struct BatchNormBaseOptions<1>;

} // namespace nn
} // namespace torch
