#include <torch/nn/options/rnn.h>

namespace torch {
namespace nn {

namespace detail {

RNNOptionsBase::RNNOptionsBase(rnn_options_base_mode_t mode, int64_t input_size, int64_t hidden_size)
    : mode_(mode), input_size_(input_size), hidden_size_(hidden_size) {}

} // namespace detail

RNNOptions::RNNOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

LSTMOptions::LSTMOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

GRUOptions::GRUOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

} // namespace nn
} // namespace torch
