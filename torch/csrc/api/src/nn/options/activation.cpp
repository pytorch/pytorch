#include <torch/nn/options/activation.h>

namespace torch {
namespace nn {

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

} // namespace nn
} // namespace torch
