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
  /// The activation to use after linear operations.
  TORCH_ARG(RNNActivation, activation) = RNNActivation::ReLU;
};

using LSTMOptions = detail::RNNOptionsBase;
using GRUOptions = detail::RNNOptionsBase;

namespace detail {

/// Common options for RNNCell, LSTMCell and GRUCell modules
struct TORCH_API RNNCellOptionsBase {
  RNNCellOptionsBase(int64_t input_size, int64_t hidden_size, bool bias, int64_t num_chunks);
  virtual ~RNNCellOptionsBase() = default;
  
  TORCH_ARG(int64_t, input_size);
  TORCH_ARG(int64_t, hidden_size);
  TORCH_ARG(bool, bias);
  TORCH_ARG(int64_t, num_chunks);
};

} // namespace detail

/// Options for RNNCell modules.
struct TORCH_API RNNCellOptions {
  typedef c10::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  RNNCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// The non-linearity to use. Can be either ``torch::kTanh`` or ``torch::kReLU``. Default: ``torch::kTanh``
  TORCH_ARG(nonlinearity_t, nonlinearity) = torch::kTanh;
};

/// Options for LSTMCell modules.
struct TORCH_API LSTMCellOptions {
  LSTMCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
};

/// Options for GRUCell modules.
struct TORCH_API GRUCellOptions {
  GRUCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
};

} // namespace nn
} // namespace torch
