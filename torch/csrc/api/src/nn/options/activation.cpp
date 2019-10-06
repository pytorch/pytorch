#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn {

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

SoftmaxOptions::SoftmaxOptions(int dim, torch::Dtype dtype)
  : dim_(dim), dtype_(dtype) {}

} // namespace nn
} // namespace torch
