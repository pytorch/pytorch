#include <torch/nn/options/activation.h>

namespace torch {
namespace nn {

SELUOptions::SELUOptions(bool inplace) : inplace_(inplace) {}

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

} // namespace nn
} // namespace torch
