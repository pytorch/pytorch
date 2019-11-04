#pragma once

#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

namespace detail {

/// Common options for LSTM and GRU modules.
struct TORCH_API RNNOptionsBase {
  RNNOptionsBase(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`.
  TORCH_ARG(int64_t, input_size);

  /// The number of features in the hidden state `h`.
  TORCH_ARG(int64_t, hidden_size);

  /// Number of recurrent layers. E.g., setting `num_layers=2`
  /// would mean stacking two RNNs together to form a `stacked RNN`,
  /// with the second RNN taking in outputs of the first RNN and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;

  /// If `false`, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: `true`
  TORCH_ARG(bool, bias) = true;

  /// If true, the input sequence should be provided as `(batch, sequence,
  /// features)`. If false (default), the expected layout is `(sequence, batch,
  /// features)`. Default: `false`
  TORCH_ARG(bool, batch_first) = false;

  /// If non-zero, introduces a `Dropout` layer on the outputs of each RNN
  /// layer except the last layer, with dropout probability equal to `dropout`.
  /// Default: 0
  TORCH_ARG(double, dropout) = 0.0;

  /// If `true`, becomes bidirectional. Default: `false`
  TORCH_ARG(bool, bidirectional) = false;
};

} // namespace detail

/// Options for RNN modules.
struct TORCH_API RNNOptions {
  typedef c10::variant<enumtype::kReLU, enumtype::kTanh> nonlinearity_t;

  RNNOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`.
  TORCH_ARG(int64_t, input_size);

  /// The number of features in the hidden state `h`.
  TORCH_ARG(int64_t, hidden_size);

  /// Number of recurrent layers. E.g., setting `num_layers=2`
  /// would mean stacking two RNNs together to form a `stacked RNN`,
  /// with the second RNN taking in outputs of the first RNN and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;

  /// If `false`, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: `true`
  TORCH_ARG(bool, bias) = true;

  /// If true, the input sequence should be provided as `(batch, sequence,
  /// features)`. If false (default), the expected layout is `(sequence, batch,
  /// features)`. Default: `false`
  TORCH_ARG(bool, batch_first) = false;

  /// If non-zero, introduces a `Dropout` layer on the outputs of each RNN
  /// layer except the last layer, with dropout probability equal to `dropout`.
  /// Default: 0
  TORCH_ARG(double, dropout) = 0.0;

  /// If `true`, becomes bidirectional. Default: `false`
  TORCH_ARG(bool, bidirectional) = false;

  /// The non-linearity to use. Can be either `Tanh` or `ReLU`. Default: `Tanh`
  TORCH_ARG(nonlinearity_t, nonlinearity) = torch::kTanh;
};

using LSTMOptions = detail::RNNOptionsBase;
using GRUOptions = detail::RNNOptionsBase;

} // namespace nn
} // namespace torch
