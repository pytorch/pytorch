#include <torch/nn/modules/rnn.h>

#include <torch/nn/modules/dropout.h>
#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/util/Exception.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace detail {
template <typename Derived>
RNNImplBase<Derived>::RNNImplBase(
    const RNNOptionsBase& options_,
    optional<CuDNNMode> cudnn_mode,
    int64_t number_of_gates)
    : options(options_),
      number_of_gates_(number_of_gates),
      cudnn_mode_(std::move(cudnn_mode)) {
  reset();
}

template <typename Derived>
void RNNImplBase<Derived>::reset() {
  const auto num_directions = options.bidirectional() ? 2 : 1;

  w_ih.resize(options.layers() * num_directions);
  w_hh.resize(options.layers() * num_directions);
  b_ih.resize(options.layers() * num_directions);
  b_hh.resize(options.layers() * num_directions);

  const int64_t gate_size = options.hidden_size() * number_of_gates_;

  for (int64_t layer = 0; layer < options.layers(); ++layer) {
    for (auto direction = 0; direction < num_directions; direction++) {
      const auto layer_input_size = layer == 0 ? options.input_size() :
        options.hidden_size() * num_directions;
      const auto suffix = direction == 1 ? "_reverse" : "";
      const auto layer_idx = (layer * num_directions) + direction;
      w_ih[layer_idx] = this->register_parameter(
          "weight_ih_l" + std::to_string(layer) + suffix,
          torch::empty({gate_size, layer_input_size}));
      w_hh[layer_idx] = this->register_parameter(
          "weight_hh_l" + std::to_string(layer) + suffix,
          torch::empty({gate_size, options.hidden_size()}));

      if (options.with_bias()) {
        b_ih[layer_idx] = this->register_parameter(
          "bias_ih_l" + std::to_string(layer) + suffix,
          torch::empty({gate_size}));
        b_hh[layer_idx] = this->register_parameter(
          "bias_hh_l" + std::to_string(layer) + suffix,
          torch::empty({gate_size}));
      }
    }
  }

  {
    NoGradGuard no_grad;
    const auto stdv = 1.0 / std::sqrt(options.hidden_size());
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
  const auto num_directions = options.bidirectional() ? 2 : 1;
  for (int64_t layer = 0; layer < options.layers(); layer++) {
    for (auto direction = 0; direction < num_directions; direction++) {
      const auto layer_idx = (layer * num_directions) + direction;
      w_ih[layer_idx] = w_ih[layer_idx].to(device, non_blocking);
      w_hh[layer_idx] = w_hh[layer_idx].to(device, non_blocking);
      if (options.with_bias()) {
        b_ih[layer_idx] = b_ih[layer_idx].to(device, non_blocking);
        b_hh[layer_idx] = b_hh[layer_idx].to(device, non_blocking);
      }
    }
  }
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::pretty_print(std::ostream& stream) const {
  const std::string name = this->name();
  const std::string name_without_impl = name.substr(0, name.size() - 4);
  stream << name_without_impl << "(input_size=" << options.input_size()
         << ", hidden_size=" << options.hidden_size()
         << ", layers=" << options.layers() << ", dropout=" << options.dropout()
         << ")";
}

template <typename Derived>
void RNNImplBase<Derived>::flatten_parameters() {
  // Cache the flattened weight and bias vector.
  flat_weights_ = flat_weights();

  if (!cudnn_mode_ || !torch::cudnn_is_acceptable(w_ih.at(0))) {
    return;
  }

  NoGradGuard no_grad;
  if (torch::_use_cudnn_rnn_flatten_weight()) {
    torch::_cudnn_rnn_flatten_weight(
        flat_weights_,
        /*weight_stride0=*/options.with_bias() ? 4 : 2,
        options.input_size(),
        static_cast<int64_t>(*cudnn_mode_),
        options.hidden_size(),
        options.layers(),
        /*batch_first=*/options.batch_first(),
        /*bidirectional=*/options.bidirectional());
  }
}

template <typename Derived>
RNNOutput RNNImplBase<Derived>::generic_forward(
    std::function<RNNFunctionSignature> function,
    const Tensor& input,
    Tensor state) {
  if (!state.defined()) {
    // #layers, batch size, state size
    const auto batch_size = input.size(options.batch_first() ? 0 : 1);
    const auto num_directions = options.bidirectional() ? 2 : 1;
    state = torch::zeros(
      {options.layers() * num_directions, batch_size, options.hidden_size()},
      input.options());
  }
  Tensor output, new_state;
  std::tie(output, new_state) = function(
      input,
      std::move(state),
      flat_weights_,
      options.with_bias(),
      options.layers(),
      options.dropout(),
      this->is_training(),
      options.bidirectional(),
      options.batch_first());
  return {output, new_state};
}

template <typename Derived>
std::vector<Tensor> RNNImplBase<Derived>::flat_weights() const {
  // Organize all weights in a flat vector in the order
  // (w_ih, w_hh, b_ih, b_hh), repeated for each layer (next to each other).
  std::vector<Tensor> flat;
  const auto num_directions = options.bidirectional() ? 2 : 1;
  for (int64_t layer = 0; layer < options.layers(); layer++) {
    for (auto direction = 0; direction < num_directions; direction++) {
      const auto layer_idx = (layer * num_directions) + direction;
      flat.push_back(w_ih[layer_idx]);
      flat.push_back(w_hh[layer_idx]);
      if (options.with_bias()) {
        flat.push_back(b_ih[layer_idx]);
        flat.push_back(b_hh[layer_idx]);
      }
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

RNNImpl::RNNImpl(const RNNOptions& options_)
    : detail::RNNImplBase<RNNImpl>(
          detail::RNNOptionsBase(options_.input_size(), options_.hidden_size())
              .layers(options_.layers())
              .with_bias(options_.with_bias())
              .dropout(options_.dropout())
              .bidirectional(options_.bidirectional())
              .batch_first(options_.batch_first()),
          static_cast<CuDNNMode>(options_.activation())),
      options(options_) {}

void RNNImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::RNN(input_size=" << options.input_size()
         << ", hidden_size=" << options.hidden_size()
         << ", layers=" << options.layers() << ", dropout=" << options.dropout()
         << ", activation="
         << (options.activation() == RNNActivation::Tanh ? "tanh" : "relu")
         << ")";
}

RNNOutput RNNImpl::forward(const Tensor& input, Tensor state) {
  switch (options.activation()) {
    case RNNActivation::ReLU:
      return generic_forward(
          static_cast<RNNFunctionSignature*>(&torch::rnn_relu),
          input,
          std::move(state));
    case RNNActivation::Tanh:
      return generic_forward(
          static_cast<RNNFunctionSignature*>(&torch::rnn_tanh),
          input,
          std::move(state));
    default:
      AT_ERROR("Unhandled RNN activation function!");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMImpl::LSTMImpl(const LSTMOptions& options_)
    : detail::RNNImplBase<LSTMImpl>(
          options_,
          CuDNNMode::LSTM,
          /*number_of_gates=*/4) {}

RNNOutput LSTMImpl::forward(const Tensor& input, Tensor state) {
  // It would be trickier to adapt the `generic_forward` for the LSTM because
  // its output has a different dimensionality (3-tuple vs. 2-tuple), while we
  // always return one state variable (stacking the hidden/cell state into one),
  // which also makes the state variables going into the `generic_forward`, and
  // the way we default-initialize the state when it is not passed, slightly
  // different. So we just re-implement it specifically for the LSTM here.
  if (!state.defined()) {
    // 2 for hidden state and cell state, then #layers, batch size, state size
    const auto batch_size = input.size(options.batch_first() ? 0 : 1);
    const auto num_directions = options.bidirectional() ? 2 : 1;
    state = torch::zeros(
        {2, options.layers() * num_directions, batch_size, options.hidden_size()},
        input.options());
  }
  Tensor output, hidden_state, cell_state;
  std::tie(output, hidden_state, cell_state) = torch::lstm(
      input,
      {state[0], state[1]},
      flat_weights_,
      options.with_bias(),
      options.layers(),
      options.dropout(),
      this->is_training(),
      options.bidirectional(),
      options.batch_first());
  return {output, torch::stack({hidden_state, cell_state})};
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUImpl::GRUImpl(const GRUOptions& options_)
    : detail::RNNImplBase<GRUImpl>(
          options_,
          CuDNNMode::GRU,
          /*number_of_gates=*/3) {}

RNNOutput GRUImpl::forward(const Tensor& input, Tensor state) {
  return generic_forward(
      static_cast<RNNFunctionSignature*>(&torch::gru), input, std::move(state));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCellImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
template <typename Derived>
RNNCellImplBase<Derived>::RNNCellImplBase(
  const RNNCellOptionsBase& options_)
  : options_base(options_) {
  reset();
}

template <typename Derived>
void RNNCellImplBase<Derived>::reset() {
  weight_ih = this->register_parameter(
    "weight_ih", torch::empty({options_base.num_chunks() * options_base.hidden_size(), options_base.input_size()}));
  weight_hh = this->register_parameter(
    "weight_hh", torch::empty({options_base.num_chunks() * options_base.hidden_size(), options_base.hidden_size()}));

  if (options_base.bias()) {
    bias_ih = this->register_parameter("bias_ih", torch::empty({options_base.num_chunks() * options_base.hidden_size()}));
    bias_hh = this->register_parameter("bias_hh", torch::empty({options_base.num_chunks() * options_base.hidden_size()}));
  } else {
    bias_ih = this->register_parameter("bias_ih", Tensor(), /*requires_grad=*/false);
    bias_hh = this->register_parameter("bias_hh", Tensor(), /*requires_grad=*/false);
  }

  reset_parameters();
}

template <typename Derived>
void RNNCellImplBase<Derived>::reset_parameters() {
  const double stdv = 1.0 / std::sqrt(options_base.hidden_size());
  for (auto& weight : this->parameters()) {
    init::uniform_(weight, -stdv, stdv);
  }
}

template <typename Derived>
void RNNCellImplBase<Derived>::pretty_print(std::ostream& stream) const {
  const std::string name = this->name();
  const std::string name_without_impl = name.substr(0, name.size() - 4);
  stream << name_without_impl
         << "(" << options_base.input_size()
         << ", " << options_base.hidden_size();
  if (!options_base.bias()) {
    stream << ", bias=" << std::boolalpha << false;
  }
  auto nonlinearity_str = this->get_nonlinearity_str();
  if (!nonlinearity_str.empty() && nonlinearity_str != "kTanh") {
    stream << ", nonlinearity=" << nonlinearity_str;
  }
  stream << ")"; 
}

template <typename Derived>
void RNNCellImplBase<Derived>::check_forward_input(const Tensor& input) const {
  TORCH_CHECK(
    input.size(1) == options_base.input_size(), 
    "input has inconsistent input_size: got ", input.size(1), " expected ", options_base.input_size());
}

template <typename Derived>
void RNNCellImplBase<Derived>::check_forward_hidden(const Tensor& input, const Tensor& hx, std::string hidden_label) const {
  TORCH_CHECK(
    input.size(0) == hx.size(0),
    "Input batch size ", input.size(0), " doesn't match hidden", hidden_label, " batch size ", hx.size(0));

  TORCH_CHECK(
    hx.size(1) == options_base.hidden_size(),
    "hidden", hidden_label, " has inconsistent hidden_size: got ", hx.size(1), ", expected ", options_base.hidden_size());
}

template <typename Derived>
std::string RNNCellImplBase<Derived>::get_nonlinearity_str() const {
  return "";
}

template class RNNCellImplBase<LSTMCellImpl>;
template class RNNCellImplBase<GRUCellImpl>;
template class RNNCellImplBase<RNNCellImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RNNCellImpl::RNNCellImpl(const RNNCellOptions& options_)
    : detail::RNNCellImplBase<RNNCellImpl>(
          detail::RNNCellOptionsBase(
            options_.input_size(),
            options_.hidden_size(),
            options_.bias(),
            /*num_chunks=*/1)),
      options(options_) {}


Tensor RNNCellImpl::forward(const Tensor& input, Tensor hx) {
  this->check_forward_input(input);
  if (!hx.defined()) {
    hx = torch::zeros({input.size(0), options.hidden_size()}, torch::dtype(input.dtype()).device(input.device()));
  }
  this->check_forward_hidden(input, hx, "");
  Tensor ret;
  if (c10::get_if<enumtype::kTanh>(&options.nonlinearity())) {
    ret = torch::rnn_tanh_cell(
      input, hx,
      weight_ih, weight_hh,
      bias_ih, bias_hh
    );
  } else if (c10::get_if<enumtype::kReLU>(&options.nonlinearity())) {
    ret = torch::rnn_relu_cell(
      input, hx,
      weight_ih, weight_hh,
      bias_ih, bias_hh
    );
  } else {
    TORCH_CHECK(false, "Unknown nonlinearity: ", torch::enumtype::get_enum_name(options.nonlinearity()));
  }
  return ret;
}

std::string RNNCellImpl::get_nonlinearity_str() const {
  return get_enum_name(options.nonlinearity());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTMCell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMCellImpl::LSTMCellImpl(const LSTMCellOptions& options_)
    : detail::RNNCellImplBase<LSTMCellImpl>(
          detail::RNNCellOptionsBase(
            options_.input_size(),
            options_.hidden_size(),
            options_.bias(),
            /*num_chunks=*/4)),
      options(options_) {}

std::tuple<Tensor, Tensor> LSTMCellImpl::forward(
  const Tensor& input, torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  this->check_forward_input(input);

  std::tuple<Tensor, Tensor> hx;
  if (!hx_opt.has_value()) {
    auto zeros = torch::zeros({input.size(0), options.hidden_size()}, torch::dtype(input.dtype()).device(input.device()));
    hx = std::make_tuple(zeros, zeros);
  } else {
    hx = hx_opt.value();
  }

  this->check_forward_hidden(input, std::get<0>(hx), "[0]");
  this->check_forward_hidden(input, std::get<1>(hx), "[1]");

  return torch::lstm_cell(
    input, {std::get<0>(hx), std::get<1>(hx)},
    weight_ih, weight_hh,
    bias_ih, bias_hh
  );
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRUCell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUCellImpl::GRUCellImpl(const GRUCellOptions& options_)
    : detail::RNNCellImplBase<GRUCellImpl>(
          detail::RNNCellOptionsBase(
            options_.input_size(),
            options_.hidden_size(),
            options_.bias(),
            /*num_chunks=*/3)),
      options(options_) {}

Tensor GRUCellImpl::forward(const Tensor& input, Tensor hx) {
  this->check_forward_input(input);
  if (!hx.defined()) {
    hx = torch::zeros({input.size(0), options.hidden_size()}, torch::dtype(input.dtype()).device(input.device()));
  }
  this->check_forward_hidden(input, hx, "");
  return torch::gru_cell(
    input, hx,
    weight_ih, weight_hh,
    bias_ih, bias_hh
  );
}

} // namespace nn
} // namespace torch
