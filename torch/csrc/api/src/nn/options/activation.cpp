#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn {

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

SoftmaxOptions::SoftmaxOptions(int dim) : dim_(dim) {}

} // namespace nn
} // namespace torch
