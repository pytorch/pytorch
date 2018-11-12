#include <torch/nn/modules/rnn.h>

#include <torch/nn/modules/dropout.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/util/Exception.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNOptionsBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
RNNOptionsBase::RNNOptionsBase(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Derived>
RNNImplBase<Derived>::RNNImplBase(
    RNNOptionsBase options_,
    optional<CuDNNMode> cudnn_mode,
    int64_t number_of_gates)
    : options(options_),
      number_of_gates_(number_of_gates),
      cudnn_mode_(std::move(cudnn_mode)) {
  reset();
}

template <typename Derived>
void RNNImplBase<Derived>::reset() {
  w_ih.resize(options.layers_);
  w_hh.resize(options.layers_);
  b_ih.resize(options.layers_);
  b_hh.resize(options.layers_);

  const int64_t gate_size = options.hidden_size_ * number_of_gates_;

  for (int64_t layer = 0; layer < options.layers_; ++layer) {
    const int64_t input_size =
        (layer == 0) ? options.input_size_ : options.hidden_size_;
    w_ih[layer] = this->register_parameter(
        "weight_ih_l" + std::to_string(layer),
        torch::empty({gate_size, input_size}));
    w_hh[layer] = this->register_parameter(
        "weight_hh_l" + std::to_string(layer),
        torch::empty({gate_size, options.hidden_size_}));

    if (options.with_bias_) {
      b_ih[layer] = this->register_parameter(
          "bias_ih_l" + std::to_string(layer), torch::empty({gate_size}));
      b_hh[layer] = this->register_parameter(
          "bias_hh_l" + std::to_string(layer), torch::empty({gate_size}));
    }
  }

  {
    NoGradGuard no_grad;
    const auto stdv = 1.0 / std::sqrt(options.hidden_size_);
    for (auto& p : this->parameters()) {
      p.uniform_(-stdv, stdv);
    }
  }

  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(
    torch::Device device,
    torch::Dtype dtype,
    bool non_blocking) {
  nn::Module::to(device, dtype, non_blocking);
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Dtype dtype, bool non_blocking) {
  nn::Module::to(dtype, non_blocking);
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Device device, bool non_blocking) {
  nn::Module::to(device, non_blocking);
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::flatten_parameters() {
  // Cache the flattened weight and bias vector.
  flat_weights_ = flat_weights();

  if (!cudnn_mode_ || !torch::cudnn_is_acceptable(w_ih.at(0))) {
    return;
  }

  NoGradGuard no_grad;
  torch::_cudnn_rnn_flatten_weight(
      flat_weights_,
      /*weight_stride0=*/options.with_bias_ ? 4 : 2,
      options.input_size_,
      static_cast<int64_t>(*cudnn_mode_),
      options.hidden_size_,
      options.layers_,
      /*batch_first=*/options.batch_first_,
      /*bidirectional=*/options.bidirectional_);
}

template <typename Derived>
RNNOutput RNNImplBase<Derived>::generic_forward(
    std::function<RNNFunctionSignature> function,
    Tensor input,
    Tensor state) {
  if (!state.defined()) {
    // #layers, batch size, state size
    const auto batch_size = input.size(options.batch_first_ ? 0 : 1);
    state = torch::zeros(
        {options.layers_, batch_size, options.hidden_size_}, input.options());
  }
  Tensor output, new_state;
  std::tie(output, new_state) = function(
      std::move(input),
      std::move(state),
      flat_weights_,
      options.with_bias_,
      options.layers_,
      options.dropout_,
      this->is_training(),
      options.bidirectional_,
      options.batch_first_);
  return {output, new_state};
}

template <typename Derived>
std::vector<Tensor> RNNImplBase<Derived>::flat_weights() const {
  // Organize all weights in a flat vector in the order
  // (w_ih, w_hh, b_ih, b_hh), repeated for each layer (next to each other).
  std::vector<Tensor> flat;
  for (int64_t layer = 0; layer < options.layers_; layer++) {
    flat.push_back(w_ih[layer]);
    flat.push_back(w_hh[layer]);
    if (options.with_bias_) {
      flat.push_back(b_ih[layer]);
      flat.push_back(b_hh[layer]);
    }
  }
  return flat;
}

template <typename Derived>
bool RNNImplBase<Derived>::any_parameters_alias() const {
  // If any parameters alias, we fall back to the slower, copying code path.
  // This is a sufficient check, because overlapping parameter buffers that
  // don't completely alias would break the assumptions of the uniqueness check
  // in Module.named_parameters().
  std::unordered_set<void*> unique_data_ptrs;
  auto params = this->parameters();
  unique_data_ptrs.reserve(params.size());
  for (const auto& p : params) {
    unique_data_ptrs.emplace(p.data_ptr());
  }
  return unique_data_ptrs.size() != params.size();
}

template class RNNImplBase<LSTMImpl>;
template class RNNImplBase<GRUImpl>;
template class RNNImplBase<RNNImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RNNOptions::RNNOptions(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

RNNOptions& RNNOptions::tanh() {
  return activation(RNNActivation::Tanh);
}

RNNOptions& RNNOptions::relu() {
  return activation(RNNActivation::ReLU);
}

RNNImpl::RNNImpl(RNNOptions options)
    : detail::RNNImplBase<RNNImpl>(
          detail::RNNOptionsBase(options.input_size_, options.hidden_size_)
              .layers(options.layers_)
              .with_bias(options.with_bias_)
              .dropout(options.dropout_)
              .bidirectional(options.bidirectional_)
              .batch_first(options.batch_first_),
          static_cast<CuDNNMode>(options.activation_)),
      options(options) {}

RNNOutput RNNImpl::forward(Tensor input, Tensor state) {
  switch (options.activation_) {
    case RNNActivation::ReLU:
      return generic_forward(
          static_cast<RNNFunctionSignature*>(&torch::rnn_relu), input, state);
    case RNNActivation::Tanh:
      return generic_forward(
          static_cast<RNNFunctionSignature*>(&torch::rnn_tanh), input, state);
    default:
      AT_ERROR("Unhandled RNN activation function!");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMImpl::LSTMImpl(LSTMOptions options)
    : detail::RNNImplBase<LSTMImpl>(
          options,
          CuDNNMode::LSTM,
          /*number_of_gates=*/4) {}

RNNOutput LSTMImpl::forward(Tensor input, Tensor state) {
  // It would be trickier to adapt the `generic_forward` for the LSTM because
  // its output has a different dimensionality (3-tuple vs. 2-tuple), while we
  // always return one state variable (stacking the hidden/cell state into one),
  // which also makes the state variables going into the `generic_forward`, and
  // the way we default-initialize the state when it is not passed, slightly
  // different. So we just re-implement it specifically for the LSTM here.
  if (!state.defined()) {
    // 2 for hidden state and cell state, then #layers, batch size, state size
    const auto batch_size = input.size(options.batch_first_ ? 0 : 1);
    state = torch::zeros(
        {2, options.layers_, batch_size, options.hidden_size_},
        input.options());
  }
  Tensor output, hidden_state, cell_state;
  std::tie(output, hidden_state, cell_state) = torch::lstm(
      input,
      {state[0], state[1]},
      flat_weights_,
      options.with_bias_,
      options.layers_,
      options.dropout_,
      this->is_training(),
      options.bidirectional_,
      options.batch_first_);
  return {output, torch::stack({hidden_state, cell_state})};
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUImpl::GRUImpl(GRUOptions options)
    : detail::RNNImplBase<GRUImpl>(
          options,
          CuDNNMode::GRU,
          /*number_of_gates=*/3) {}

RNNOutput GRUImpl::forward(Tensor input, Tensor state) {
  return generic_forward(
      static_cast<RNNFunctionSignature*>(&torch::gru), input, state);
}
} // namespace nn
} // namespace torch
