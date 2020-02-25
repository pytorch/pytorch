#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

namespace detail {

/// Common options for LSTM and GRU modules.
struct TORCH_API RNNOptionsBase {
  RNNOptionsBase(int64_t input_size, int64_t hidden_size);
  virtual ~RNNOptionsBase() = default;
  /// The number of features of a single sample in the input sequence `x`.
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`.
  TORCH_ARG(int64_t, hidden_size);
  /// The number of recurrent layers (cells) to use.
  TORCH_ARG(int64_t, layers) = 1;
  /// Whether a bias term should be added to all linear operations.
  TORCH_ARG(bool, with_bias) = true;
  /// If non-zero, adds dropout with the given probability to the output of each
  /// RNN layer, except the final layer.
  TORCH_ARG(double, dropout) = 0.0;
  /// Whether to make the RNN bidirectional.
  TORCH_ARG(bool, bidirectional) = false;
  /// If true, the input sequence should be provided as `(batch, sequence,
  /// features)`. If false (default), the expected layout is `(sequence, batch,
  /// features)`.
  TORCH_ARG(bool, batch_first) = false;
  // If false, and bidirectionality is enabled, the RNN will not concatenate both
  // forward and backward hidden states for each layer output, instead forward
  // and backward states will be processed independently and then concatenated
  // at the final output of the network. If true then both forward and
  // backward states will be concatenated after each layer.
  TORCH_ARG(bool, cat_layer_fwd_bwd_states) = true;
};

} // namespace detail

enum class RNNActivation : uint32_t {ReLU, Tanh};

/// Options for RNN modules.
struct TORCH_API RNNOptions {
  RNNOptions(int64_t input_size, int64_t hidden_size);

  /// Sets the activation after linear operations to `tanh`.
  RNNOptions& tanh();
  /// Sets the activation after linear operations to `relu`.
  RNNOptions& relu();

  /// The number of features of a single sample in the input sequence `x`.
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`.
  TORCH_ARG(int64_t, hidden_size);
  /// The number of recurrent layers (cells) to use.
  TORCH_ARG(int64_t, layers) = 1;
  /// Whether a bias term should be added to all linear operations.
  TORCH_ARG(bool, with_bias) = true;
  /// If non-zero, adds dropout with the given probability to the output of each
  /// RNN layer, except the final layer.
  TORCH_ARG(double, dropout) = 0.0;
  /// Whether to make the RNN bidirectional.
  TORCH_ARG(bool, bidirectional) = false;
  /// If true, the input sequence should be provided as `(batch, sequence,
  /// features)`. If false (default), the expected layout is `(sequence, batch,
  /// features)`.
  TORCH_ARG(bool, batch_first) = false;
  // If false, and bidirectionality is enabled, the RNN will not concatenate both
  // forward and backward hidden states for each layer output, instead forward
  // and backward states will be processed independently and then concatenated
  // at the final output of the network. If true then both forward and
  // backward states will be concatenated after each layer.
  TORCH_ARG(bool, cat_layer_fwd_bwd_states) = true;
  /// The activation to use after linear operations.
  TORCH_ARG(RNNActivation, activation) = RNNActivation::ReLU;
};

using LSTMOptions = detail::RNNOptionsBase;
using GRUOptions = detail::RNNOptionsBase;

} // namespace nn
} // namespace torch
