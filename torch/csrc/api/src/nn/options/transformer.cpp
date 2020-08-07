#include <torch/nn/options/transformer.h>

namespace torch {
namespace nn {

TransformerEncoderLayerOptions::TransformerEncoderLayerOptions(
  int64_t d_model, int64_t nhead) : d_model_(d_model), nhead_(nhead) {}


} // namespace nn
} // namespace torch
