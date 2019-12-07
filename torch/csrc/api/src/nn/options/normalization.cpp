#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {

LayerNormOptions::LayerNormOptions(std::vector<int64_t> normalized_shape) : normalized_shape_(std::move(normalized_shape)) {}

CrossMapLRN2dOptions::CrossMapLRN2dOptions(int64_t size) : size_(size) {}

GroupNormOptions::GroupNormOptions(int64_t num_groups, int64_t num_channels)
  : num_groups_(num_groups), num_channels_(num_channels) {}

namespace functional {

LayerNormFuncOptions::LayerNormFuncOptions(std::vector<int64_t> normalized_shape) : normalized_shape_(std::move(normalized_shape)) {}

GroupNormFuncOptions::GroupNormFuncOptions(int64_t num_groups) : num_groups_(num_groups) {}

} // namespace functional

} // namespace nn
} // namespace torch
