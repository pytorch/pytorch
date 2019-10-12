#include <torch/nn/options/batchnorm.h>

namespace torch {
namespace nn {

BatchNormOptions::BatchNormOptions(int64_t features) : features_(features) {}

} // namespace nn
} // namespace torch
