#include <torch/nn/modules/rnn.h>

#include <torch/nn/modules/dropout.h>

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch { namespace nn {
namespace {
Variable linear(Tensor x, Tensor w, Tensor b) {
  if (x.ndimension() == 2 && b.defined()) {
    // Fused op is marginally faster
    assert(x.size(1) == w.size(1));
    return at::addmm(b, x, w.t());
  }

  auto output = x.matmul(w.t());
  if (b.defined()) {
    output += b;
  }
  return output;
}
} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Derived>
RNNBase<Derived>::RNNBase(
    int64_t input_size,
    int64_t hidden_size,
    at::optional<CuDNNMode> cudnn_mode,
    int64_t number_of_gates,
    bool has_cell_state)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      number_of_gates_(number_of_gates),
      has_cell_state_(has_cell_state),
      cudnn_mode_(std::move(cudnn_mode)) {}

template <typename Derived>
void RNNBase<Derived>::reset() {
  if (dropout_ > 0.0) {
    dropout_module_ = Dropout(dropout_).build();
  }

  const int64_t gate_size = hidden_size_ * number_of_gates_;

  for (int64_t layer = 0; layer < layers_; ++layer) {
    const int64_t input_size = (layer == 0) ? input_size_ : hidden_size_;
    ihw_.push_back(this->add(
        Var(at::CPU(at::kFloat).empty({gate_size, input_size})),
        "weight_ih_l" + std::to_string(layer)));
    hhw_.push_back(this->add(
        Var(at::CPU(at::kFloat).empty({gate_size, hidden_size_})),
        "weight_hh_l" + std::to_string(layer)));
    if (with_bias_) {
      ihb_.push_back(this->add(
          Var(at::CPU(at::kFloat).empty({gate_size})),
          "bias_ih_l" + std::to_string(layer)));
      hhb_.push_back(this->add(
          Var(at::CPU(at::kFloat).empty({gate_size})),
          "bias_hh_l" + std::to_string(layer)));
    } else {
      ihb_.emplace_back();
      hhb_.emplace_back();
    }
  }
  flatten_parameters_for_cudnn();

  auto stdv = 1.0 / std::sqrt(hidden_size_);
  for (auto& p : this->parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

template <typename Derived>
variable_list RNNBase<Derived>::forward(variable_list inputs) {
  variable_list inp = {inputs[0], inputs.size() > 1 ? inputs[1] : Variable()};
  if (cudnn_mode_.has_value() && at::cudnn_is_acceptable(inp[0]) &&
      dropout_ == 0) {
    return {CUDNN_forward(inp)};
  } else {
    return {autograd_forward(inp)};
  }
}

template <typename Derived>
std::vector<Tensor> RNNBase<Derived>::flat_weights() const {
  std::vector<Tensor> flat;
  for (int64_t layer = 0; layer < layers_; layer++) {
    flat.push_back(ihw_[layer]);
    flat.push_back(hhw_[layer]);
    if (with_bias_) {
      flat.push_back(ihb_[layer]);
      flat.push_back(hhb_[layer]);
    }
  }
  return flat;
}

template <typename Derived>
variable_list RNNBase<Derived>::autograd_forward(variable_list inputs) {
  auto inp = inputs[0];

  std::vector<Tensor> state;
  auto has_hidden = inputs[1].defined();
  auto layer_dimension = has_hidden ? inputs[1].ndimension() - 3 : -1;
  for (int64_t layer = 0; layer < layers_; layer++) {
    state.push_back(
        has_hidden ? inputs[1].select(layer_dimension, layer) : Variable());
  }

  auto output =
      Variable(inp.type().zeros({inp.size(0), inp.size(1), hidden_size_}));
  for (int64_t t = 0; t < inp.size(0); t++) {
    auto x = inp.select(0, t);
    for (int64_t i = 0; i < layers_; i++) {
      // cell_forward() returns a stacked tensor of one or more cell states.
      auto layer_output = cell_forward({x, state[i]}, i);
      // If there are multiple cell states, keep all. If there is only one,
      // the first dimension will be 1, so `.squeeze(0)` will unpack it.
      state[i] = layer_output[0].squeeze(0);
      // x should always be the hidden cell state h, assumed to be the zero-th.
      x = layer_output[0][0];
      output.select(0, t).copy_(x);
      if (dropout_ > 0 && i != layers_ - 1) {
        x = dropout_module_->forward({x})[0];
      }
    }
  }

  auto state_output = at::stack(state);
  if (has_cell_state_) {
    state_output.transpose_(0, 1);
  }
  return variable_list({output, state_output});
}

template <typename Derived>
void RNNBase<Derived>::flatten_parameters_for_cudnn() {
  data_ptrs_.clear();
  const auto any_parameter = ihw_[0];
  if (!cudnn_mode_.has_value() || !any_parameter.is_cuda() ||
      !at::cudnn_is_acceptable(any_parameter) || dropout_ == 0) {
    return;
  }
  std::unordered_set<void*> unique_data_ptrs;
  auto params = this->parameters();
  for (auto& p : params) {
    unique_data_ptrs.insert(p.second.data().data_ptr());
  }
  // TODO PyTorch says:
  // If any parameters alias, we fall back to the slower, copying code path.
  // This is
  // a sufficient check, because overlapping parameter buffers that don't
  // completely
  // alias would break the assumptions of the uniqueness check in
  // Module.named_parameters().
  // But I'm not sure if this is the case for us
  if (unique_data_ptrs.size() != params.size()) {
    return;
  }

  {
    no_grad_guard guard;
    flat_weights_ = at::_cudnn_rnn_flatten_weight(
        flat_weights(),
        /*weight_stride=*/with_bias_ ? 4 : 2,
        input_size_,
        static_cast<int64_t>(*cudnn_mode_),
        hidden_size_,
        layers_,
        false,
        false); // batch_first and bidirectional, unsupported
  }
  for (auto& p : params) {
    data_ptrs_.emplace_back(p.second.data().data_ptr());
  }
}

template <typename Derived>
variable_list RNNBase<Derived>::CUDNN_forward(variable_list inputs) {
  auto x = inputs[0];
  Variable hx, cx;
  if (inputs[1].defined()) {
    if (has_cell_state_) {
      hx = inputs[1][0];
      cx = inputs[1][1];
    } else {
      hx = inputs[1];
    }
  } else {
    hx = x.type().zeros({layers_, x.size(1), hidden_size_});
    if (has_cell_state_) {
      cx = x.type().zeros({layers_, x.size(1), hidden_size_});
    }
  }
  auto dropout_state = x.type().tensor();

  std::vector<void*> weight_data_ptrs;
  auto params = this->parameters();
  for (auto& p : params) {
    weight_data_ptrs.emplace_back(p.second.data().data_ptr());
  }
  if (weight_data_ptrs != data_ptrs_) {
    std::cerr
        << "Parameters are unflattened! Code path might be super slow. "
        << "Please call flatten_parameters_for_cudnn() when you muck around with "
        << "storages !" << std::endl;
    flat_weights_ = Variable();
  }

  AT_CHECK(cudnn_mode_.has_value(), "No CuDNN mode has been supplied!");

  // tup = std::tuple of output, hy, cy, reserve, new_weight_buf
  auto tup = _cudnn_rnn(
      x,
      flat_weights(),
      /*weight_stride=*/with_bias_ ? 4 : 2,
      flat_weights_,
      hx,
      cx,
      static_cast<int64_t>(*cudnn_mode_),
      hidden_size_,
      layers_,
      false, // batch first
      0, // TODO Use C++ dropout descriptors
      this->is_training(),
      false, // bidirectional
      {}, // packing not supported
      dropout_state // TODO waiting on dropout state descriptor in C++ pytorch
  );

  Variable hidden_output;
  if (has_cell_state_) {
    hidden_output = at::stack({std::get<1>(tup), std::get<2>(tup)}, 0);
  } else {
    hidden_output = std::get<1>(tup);
  }

  Variable output = std::get<0>(tup);
  return variable_list({output, hidden_output});
}

template <typename Derived>
void RNNBase<Derived>::to(at::Type& type) {
  nn::Module::to(type);
  flatten_parameters_for_cudnn();
}

template <typename Derived>
void RNNBase<Derived>::to(at::ScalarType scalar_type) {
  nn::Module::to(scalar_type);
  flatten_parameters_for_cudnn();
}

template <typename Derived>
void RNNBase<Derived>::to(at::Backend backend) {
  nn::Module::to(backend);
  flatten_parameters_for_cudnn();
}

template class RNNBase<LSTM>;
template class RNNBase<GRU>;
template class RNNBase<RNN>;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTM::LSTM(int64_t input_size, int64_t hidden_size)
    : RNNBase(
          input_size,
          hidden_size,
          CuDNNMode::LSTM,
          /*number_of_gates=*/4,
          /*has_cell_state=*/true) {}

variable_list LSTM::cell_forward(variable_list inputs, int64_t layer) {
  auto x = inputs[0];
  auto hid = inputs[1].defined() ? inputs[1]
                                 : x.type().zeros({2, x.size(0), hidden_size_});
  auto hx = hid[0];
  auto cx = hid[1];

  auto gates = linear(x, ihw_[layer], ihb_[layer]) +
      linear(hx, hhw_[layer], hhb_[layer]);

  auto chunked = gates.chunk(4, 1);
  auto in_gate = chunked[0].sigmoid();
  auto forget_gate = chunked[1].sigmoid();
  auto cell_gate = chunked[2].tanh();
  auto out_gate = chunked[3].sigmoid();

  auto cy = (forget_gate * cx) + (in_gate * cell_gate);
  auto hy = out_gate * cy.tanh();

  return {at::stack({hy, cy}, 0)};
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRU::GRU(int64_t input_size, int64_t hidden_size)
    : RNNBase(input_size, hidden_size, CuDNNMode::GRU, /*number_of_gates=*/3) {}

variable_list GRU::cell_forward(variable_list inputs, int64_t layer) {
  auto x = inputs[0];
  auto hx = inputs[1].defined() ? inputs[1]
                                : x.type().zeros({x.size(0), hidden_size_});

  auto gi = linear(x, ihw_[layer], ihb_[layer]);
  auto gh = linear(x, hhw_[layer], hhb_[layer]);
  auto gic = gi.chunk(3, 1);
  auto ghc = gh.chunk(3, 1);

  auto reset_gate = (gic[0] + ghc[0]).sigmoid_();
  auto input_gate = (gic[1] + ghc[1]).sigmoid_();
  auto new_gate = (gic[2] + reset_gate * ghc[2]).tanh_();
  auto hy = new_gate + input_gate * (hx - new_gate);

  return {at::stack(hy)};
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RNN::RNN(int64_t input_size, int64_t hidden_size)
    : RNNBase(input_size, hidden_size) {}

void RNN::reset() {
  static std::array<ActivationFunction, 2> activations = {at::relu, at::tanh};
  cudnn_mode_ = static_cast<CuDNNMode>(activation_);
  RNNBase<RNN>::reset();
  activation_function_ = activations.at(static_cast<int64_t>(activation_));
}

RNN& RNN::tanh() {
  return activation(Activation::Tanh);
}

RNN& RNN::relu() {
  return activation(Activation::ReLU);
}

variable_list RNN::cell_forward(variable_list inputs, int64_t layer) {
  auto x = inputs[0];
  auto hx = inputs[1].defined() ? inputs[1]
                                : x.type().zeros({x.size(0), hidden_size_});

  auto h = linear(x, ihw_[layer], ihb_[layer]) +
      linear(hx, hhw_[layer], hhb_[layer]);

  return {at::stack(activation_function_(h))};
}
}} // namespace torch::nn
