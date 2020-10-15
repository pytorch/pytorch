#include <torch/nn/options/transformer.h>
#include <torch/nn/options/transformerlayer.h>
#include <torch/nn/options/transformercoder.h>

namespace torch {
namespace nn {

TransformerEncoderLayerOptions::TransformerEncoderLayerOptions(
  int64_t d_model, int64_t nhead) : d_model_(d_model), nhead_(nhead) {}


TransformerDecoderLayerOptions::TransformerDecoderLayerOptions(
  int64_t d_model, int64_t nhead) : d_model_(d_model), nhead_(nhead){}


TransformerEncoderOptions::TransformerEncoderOptions(
  TransformerEncoderLayer encoder_layer, int64_t num_layers) :
  encoder_layer_(std::move(encoder_layer)), num_layers_(num_layers) {}


TransformerEncoderOptions::TransformerEncoderOptions(
  const TransformerEncoderLayerOptions& encoder_layer_options, int64_t num_layers) :
  encoder_layer_(encoder_layer_options), num_layers_(num_layers) {}


TransformerDecoderOptions::TransformerDecoderOptions(
  TransformerDecoderLayer decoder_layer, int64_t num_layers) :
  decoder_layer_(std::move(decoder_layer)), num_layers_(num_layers) {}


TransformerDecoderOptions::TransformerDecoderOptions(
  const TransformerDecoderLayerOptions& decoder_layer_options,
  int64_t num_layers)
  : decoder_layer_(decoder_layer_options), num_layers_(num_layers){}


TransformerOptions::TransformerOptions(int64_t d_model, int64_t nhead) :
  d_model_(d_model), nhead_(nhead) {}

TransformerOptions::TransformerOptions(
  int64_t d_model, int64_t nhead, int64_t num_encoder_layers, int64_t num_decoder_layers) :
  d_model_(d_model), nhead_(nhead),
  num_encoder_layers_(num_encoder_layers), num_decoder_layers_(num_decoder_layers) {}

} // namespace nn
} // namespace torch
