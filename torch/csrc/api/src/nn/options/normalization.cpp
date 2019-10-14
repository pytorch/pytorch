#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {

LayerNormOptions::LayerNormOptions(torch::IntArrayRef normalized_shape) : normalized_shape(normalized_shape) {}

} // namespace nn
} // namespace torch
