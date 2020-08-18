#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <torch/enum.h>
#include <torch/nn/module.h>

namespace torch {
namespace nn {

/// Options for the `TransformerDecoder` module.
///
/// Example:
/// ```
/// TransformerDecoderLayer decoder_layer(TransformerDecoderLayerOptions(512, 8));
/// TransformerDecoder transformer_decoder(TransformerDecoderOptions(decoder_layer).num_layers(6));
/// ```
struct TORCH_API TransformerDecoderOptions {
  TransformerDecoderOptions(TransformerDecoderLayerOptions decoder_layer, int64_t num_layers);

  //TODO: using decoderLayer is resulting in cyclic dependency
  /// decoder layer to be cloned
  TORCH_ARG(TransformerDecoderLayerOptions, decoder_layer);

  /// number of heads in the multiheadattention models
  TORCH_ARG(int64_t, num_layers);

  /// the layer normalization component
  TORCH_ARG(Module, norm);
};

} // namespace nn
} // namespace torch
