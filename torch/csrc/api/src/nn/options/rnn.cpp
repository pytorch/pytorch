#include <torch/nn/options/rnn.h>

namespace torch {
namespace nn {

namespace detail {

RNNOptionsBase::RNNOptionsBase(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

} // namespace detail

RNNOptions::RNNOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

RNNOptions& RNNOptions::tanh() {
  return activation(RNNActivation::Tanh);
}

RNNOptions& RNNOptions::relu() {
  return activation(RNNActivation::ReLU);
}

namespace detail {

RNNCellOptionsBase::RNNCellOptionsBase(int64_t input_size, int64_t hidden_size, bool bias, int64_t num_chunks)
    : input_size_(input_size), hidden_size_(hidden_size), bias_(bias), num_chunks_(num_chunks) {}

} // namespace detail

RNNCellOptions::RNNCellOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

LSTMCellOptions::LSTMCellOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

GRUCellOptions::GRUCellOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

} // namespace nn
} // namespace torch
