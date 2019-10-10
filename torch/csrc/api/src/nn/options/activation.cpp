#include <torch/nn/options/activation.h>

namespace torch {
namespace nn {

SELUOptions::SELUOptions(bool inplace) : inplace_(inplace) {}

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

SoftmaxOptions::SoftmaxOptions(int64_t dim) : dim_(dim) {}

} // namespace nn
} // namespace torch
