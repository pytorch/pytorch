#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {

LayerNormOptions::LayerNormOptions(std::vector<int64_t> normalized_shape) : normalized_shape_(normalized_shape) {}

} // namespace nn
} // namespace torch
