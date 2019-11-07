#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {

LayerNormOptions::LayerNormOptions(std::vector<int64_t> normalized_shape) : normalized_shape_(std::move(normalized_shape)) {}

CrossMapLRN2dOptions::CrossMapLRN2dOptions(int64_t size) : size_(size) {}

} // namespace nn
} // namespace torch
