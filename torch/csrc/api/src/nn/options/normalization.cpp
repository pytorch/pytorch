#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {

LayerNormOptions::LayerNormOptions(std::vector<int64_t> normalized_shape) : normalized_shape_(std::move(normalized_shape)) {}

} // namespace nn
} // namespace torch
