#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <torch/enum.h>

#include <torch/nn/modules/transformerlayer.h>
#include <torch/nn/modules/container/any.h>

namespace torch {
namespace nn {

  /// Options for the `TransformerEncoder`
  ///
  /// Example:
  /// ```
  /// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512, 8).dropout(0.1));
  /// auto options = TransformerEncoderOptions(encoderLayer, 6).norm(LayerNorm(LayerNormOptions({2})));
  /// ```
  struct TORCH_API TransformerEncoderOptions {
    // This constructor will keep a shallow copy of encoder_layer, so it keeps all the data in encoder_layer.
    TransformerEncoderOptions(TransformerEncoderLayer encoder_layer, int64_t num_layers);
    // This constructor will create a new TransformerEncoderLayer obj based on passed in encoder_layer_options.
    TransformerEncoderOptions(const TransformerEncoderLayerOptions& encoder_layer_options, int64_t num_layers);

    /// transformer Encoder Layer
    TORCH_ARG(TransformerEncoderLayer, encoder_layer) = nullptr;

    /// number of encoder layers
    TORCH_ARG(int64_t, num_layers);

    /// normalization module
    TORCH_ARG(AnyModule, norm);
  };

} // namespace nn
} // namespace torch
