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

} // namespace nn
} // namespace torch
