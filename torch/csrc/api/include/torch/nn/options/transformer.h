#pragma once

#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>


namespace torch {
namespace nn {

/// Options for the `TransformerDecoderLayer` module.
///
/// Example:
/// ```
/// TransformerDecoderLayer model(TransformerDecoderLayerOptions(512, 8).dropout(0.2));
/// ```
struct TORCH_API TransformerDecoderLayerOptions {
  typedef c10::variant<enumtype::kGELU, enumtype::kReLU> activation_t;

  TransformerDecoderLayerOptions(int64_t d_model, int64_t nhead);

  /// number of expected features in the input
  TORCH_ARG(int64_t, d_model);

  /// number of heads in the multiheadattention models
  TORCH_ARG(int64_t, nhead);

  /// dimension of the feedforward network model. Default: 2048
  TORCH_ARG(int64_t, dim_feedforward) = 2048;

  /// dropout value. Default: 1
  TORCH_ARG(double, dropout) = 0.1;

  /// activation function of intermediate layer, can be either ``torch::kGELU`` or ``torch::kReLU``. Default: ``torch::kReLU``
  TORCH_ARG(activation_t, activation) = torch::kReLU;
};


} // namespace nn
} // namespace torch
