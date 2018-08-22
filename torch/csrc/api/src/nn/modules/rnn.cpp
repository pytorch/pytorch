#include <torch/nn/modules/rnn.h>

#include <torch/nn/modules/dropout.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <ATen/core/Error.h>
#include <ATen/core/optional.h>

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
namespace {
Tensor linear(Tensor x, Tensor w, Tensor b) {
  if (x.ndimension() == 2 && b.defined()) {
    // Fused op is marginally faster
    assert(x.size(1) == w.size(1));
    return torch::addmm(b, x, w.t());
  }

  auto output = x.matmul(w.t());
  if (b.defined()) {
    output += b;
  }
  return output;
}
} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNOptionsBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
RNNOptionsBase::RNNOptionsBase(int64_t input_size, int64_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Derived>
RNNImplBase<Derived>::RNNImplBase(
    RNNOptionsBase options_,
    at::optional<CuDNNMode> cudnn_mode,
    int64_t number_of_gates,
    bool has_cell_state)
    : options(options_),
      dropout(nullptr),
      number_of_gates_(number_of_gates),
      has_cell_state_(has_cell_state),
      cudnn_mode_(cudnn_mode) {
  reset();
}

template <typename Derived>
void RNNImplBase<Derived>::reset() {
  if (options.dropout_ > 0.0) {
    dropout = Dropout(options.dropout_);
  }

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

  const auto stdv = 1.0 / std::sqrt(options.hidden_size_);
  NoGradGuard no_grad;;
  for (auto& p : this->parameters()) {
    p->uniform_(-stdv, stdv);
  }
}

template <typename Derived>
RNNOutput RNNImplBase<Derived>::forward(Tensor input, Tensor state) {
  if (use_cudnn(/*sample=*/input)) {
    return CUDNN_forward(input, state);
  } else {
    return autograd_forward(input, state);
  }
}

template <typename Derived>
std::vector<Tensor> RNNImplBase<Derived>::flat_weights() const {
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
bool RNNImplBase<Derived>::use_cudnn(Tensor sample) const {
  return cudnn_mode_.has_value() && sample.is_cuda() &&
      torch::cudnn_is_acceptable(sample);
}

template <typename Derived>
Tensor RNNImplBase<Derived>::create_dropout_state(Tensor input) const {
  static const int64_t dropout_seed =
      torch::empty({}, torch::kInt64).random_().toCLong();
  if (options.dropout_ > 0) {
    torch::DeviceGuard guard(input.device());
    return torch::_cudnn_init_dropout_state(
        input.type().toScalarType(torch::kUInt8),
        options.dropout_,
        this->is_training(),
        dropout_seed);
  }
  return torch::empty({}, input.options());
}

template <typename Derived>
RNNOutput RNNImplBase<Derived>::autograd_forward(Tensor input, Tensor state) {
  std::vector<Tensor> new_state;
  auto has_hidden = state.defined();
  auto layer_dimension = has_hidden ? state.ndimension() - 3 : -1;
  for (int64_t layer = 0; layer < options.layers_; layer++) {
    new_state.push_back(
        has_hidden ? state.select(layer_dimension, layer) : Tensor());
  }

  auto output = torch::zeros(
      {input.size(0), input.size(1), options.hidden_size_}, input.options());
  for (int64_t t = 0; t < input.size(0); t++) {
    auto x = input.select(0, t);
    for (int64_t i = 0; i < options.layers_; i++) {
      // cell_forward() returns a stacked tensor of one or more cell states.
      auto layer_output = cell_forward(x, new_state[i], i);
      // If there are multiple cell states, keep all. If there is only one,
      // the first dimension will be 1, so `.squeeze(0)` will unpack it.
      new_state[i] = layer_output.squeeze(0);
      // x should always be the hidden cell state h, assumed to be the zero-th.
      x = layer_output[0];
      output.select(0, t).copy_(x);
      if (options.dropout_ > 0 && i != options.layers_ - 1) {
        x = dropout->forward(x);
      }
    }
  }

  auto state_output = torch::stack(new_state);
  if (has_cell_state_) {
    state_output.transpose_(0, 1);
  }
  return {output, state_output};
}

template <typename Derived>
void RNNImplBase<Derived>::flatten_parameters_for_cudnn() {
  data_ptrs_.clear();
  const auto any_parameter = w_ih.at(0);
  if (!use_cudnn(/*sample=*/w_ih.at(0))) {
    return;
  }
  std::unordered_set<void*> unique_data_ptrs;
  auto params = this->parameters();
  for (auto& p : params) {
    unique_data_ptrs.insert(p->data_ptr());
  }
  // TODO PyTorch says: If any parameters alias, we fall back to the slower,
  // copying code path. This is a sufficient check, because overlapping
  // parameter buffers that don't completely alias would break the assumptions
  // of the uniqueness check in Module.named_parameters(). But I'm not sure if
  // this is the case for us
  if (unique_data_ptrs.size() != params.size()) {
    return;
  }

  {
    NoGradGuard no_grad;;
    flat_weights_ = torch::_cudnn_rnn_flatten_weight(
        flat_weights(),
        /*weight_stride=*/options.with_bias_ ? 4 : 2,
        options.input_size_,
        static_cast<int64_t>(*cudnn_mode_),
        options.hidden_size_,
        options.layers_,
        /*batch_first=*/false,
        /*bidirectional=*/false);
  }
  for (auto& p : params) {
    data_ptrs_.emplace_back(p->data_ptr());
  }
}

template <typename Derived>
RNNOutput RNNImplBase<Derived>::CUDNN_forward(Tensor input, Tensor state) {
  Tensor hx, cx;
  if (state.defined()) {
    if (has_cell_state_) {
      hx = state[0];
      cx = state[1];
    } else {
      hx = state;
    }
  } else {
    hx = torch::zeros(
        {options.layers_, input.size(1), options.hidden_size_},
        input.options());
    if (has_cell_state_) {
      cx = torch::zeros(
          {options.layers_, input.size(1), options.hidden_size_},
          input.options());
    }
  }
  std::vector<void*> weight_data_ptrs;
  for (auto& p : this->parameters()) {
    weight_data_ptrs.emplace_back(p->data_ptr());
  }

  AT_CHECK(
      weight_data_ptrs == data_ptrs_,
      "Parameters are unflattened! Code path might be super slow. "
      "Please call flatten_parameters_for_cudnn() when you muck "
      "around with storages!")

  // cudnn_output = std::tuple<output, hy, cy, reserve, new_weight_buf>
  auto cudnn_output = torch::_cudnn_rnn(
      /*input=*/input,
      /*weight=*/flat_weights(),
      /*weight_stride0=*/options.with_bias_ ? 4 : 2,
      /*weight_buf=*/flat_weights_,
      /*hx=*/hx,
      /*cx=*/cx,
      /*mode=*/static_cast<int64_t>(*cudnn_mode_),
      /*hidden_size=*/options.hidden_size_,
      /*num_layers=*/options.layers_,
      /*batch_first=*/false,
      /*dropout=*/options.dropout_,
      /*train=*/this->is_training(),
      /*bidirectional=*/false,
      /*batch_sizes=*/{},
      /*dropout_state=*/create_dropout_state(input));

  Tensor hidden_output = std::get<1>(cudnn_output);
  if (has_cell_state_) {
    auto cy = std::get<2>(cudnn_output);
    hidden_output = torch::stack({hidden_output, cy});
  }

  Tensor output = std::get<0>(cudnn_output);
  return {output, hidden_output};
}

template <typename Derived>
void RNNImplBase<Derived>::to(
    torch::Device device,
    torch::Dtype dtype,
    bool non_blocking) {
  nn::Module::to(device, dtype, non_blocking);
  flatten_parameters_for_cudnn();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Dtype dtype, bool non_blocking) {
  nn::Module::to(dtype, non_blocking);
  flatten_parameters_for_cudnn();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Device device, bool non_blocking) {
  nn::Module::to(device, non_blocking);
  flatten_parameters_for_cudnn();
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
              .dropout(options.dropout_),
          /*cudnn_mode=*/static_cast<CuDNNMode>(options.activation_)),
      options(options) {
  switch (options.activation_) {
    case RNNActivation::ReLU: {
      activation_function_ = torch::relu;
      break;
    }
    case RNNActivation::Tanh: {
      activation_function_ = torch::tanh;
      break;
    }
  }
}

Tensor RNNImpl::cell_forward(Tensor input, Tensor state, int64_t layer) {
  auto hx = state.defined()
      ? state
      : torch::zeros({input.size(0), options.hidden_size_}, input.options());

  auto h = linear(input, w_ih[layer], b_ih[layer]) +
      linear(hx, w_hh[layer], b_hh[layer]);

  return torch::stack(activation_function_(h));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMImpl::LSTMImpl(LSTMOptions options)
    : detail::RNNImplBase<LSTMImpl>(
          options,
          /*cudnn_mode=*/CuDNNMode::LSTM,
          /*number_of_gates=*/4,
          /*has_cell_state=*/true) {}

Tensor LSTMImpl::cell_forward(Tensor input, Tensor state, int64_t layer) {
  auto hid = state.defined()
      ? state
      : torch::zeros({2, input.size(0), options.hidden_size_}, input.options());
  auto hx = hid[0];
  auto cx = hid[1];

  auto gates = linear(input, w_ih[layer], b_ih[layer]) +
      linear(hx, w_hh[layer], b_hh[layer]);

  auto chunked = gates.chunk(4, 1);
  auto in_gate = chunked[0].sigmoid();
  auto forget_gate = chunked[1].sigmoid();
  auto cell_gate = chunked[2].tanh();
  auto out_gate = chunked[3].sigmoid();

  auto cy = (forget_gate * cx) + (in_gate * cell_gate);
  auto hy = out_gate * cy.tanh();

  return torch::stack({hy, cy}, 0);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUImpl::GRUImpl(GRUOptions options)
    : detail::RNNImplBase<GRUImpl>(
          options,
          /*cudnn_mode=*/CuDNNMode::GRU,
          /*number_of_gates=*/3) {}

Tensor GRUImpl::cell_forward(Tensor input, Tensor state, int64_t layer) {
  auto hx = state.defined()
      ? state
      : torch::zeros({input.size(0), options.hidden_size_}, input.options());

  auto gi = linear(input, w_ih[layer], b_ih[layer]);
  auto gh = linear(input, w_hh[layer], b_hh[layer]);
  auto gic = gi.chunk(3, 1);
  auto ghc = gh.chunk(3, 1);

  auto reset_gate = (gic[0] + ghc[0]).sigmoid_();
  auto input_gate = (gic[1] + ghc[1]).sigmoid_();
  auto new_gate = (gic[2] + reset_gate * ghc[2]).tanh_();
  auto hy = new_gate + input_gate * (hx - new_gate);

  return torch::stack(hy);
}
} // namespace nn
} // namespace torch
