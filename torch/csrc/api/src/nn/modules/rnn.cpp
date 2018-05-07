#include <torch/nn/modules/rnn.h>

#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>

#include <cmath>
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

template <typename Derived>
void RNNBase<Derived>::initialize_containers() {
  if (dropout_ > 0)
    dropout_module = make(Dropout(dropout_));
}

template <typename Derived>
void RNNBase<Derived>::initialize_parameters() {
  auto gate_size = hidden_size_;
  if (mode_ == RNNMode::LSTM) {
    gate_size *= 4;
  } else if (mode_ == RNNMode::GRU) {
    gate_size *= 3;
  }

  ihw.clear();
  hhw.clear();
  ihb.clear();
  hhb.clear();
  for (auto i = 0U; i < nlayers_; i++) {
    auto input_size = (i == 0) ? input_size_ : hidden_size_;
    ihw.push_back(this->add(
        Var(this->DefaultTensor(at::kFloat).tensor({gate_size, input_size})),
        "weight_ih_l" + std::to_string(i)));
    hhw.push_back(this->add(
        Var(this->DefaultTensor(at::kFloat).tensor({gate_size, hidden_size_})),
        "weight_hh_l" + std::to_string(i)));
    ihb.push_back(
        no_bias_ ? Variable()
                 : this->add(
                       Var(this->DefaultTensor(at::kFloat).tensor({gate_size})),
                       "bias_ih_l" + std::to_string(i)));
    hhb.push_back(
        no_bias_ ? Variable()
                 : this->add(
                       Var(this->DefaultTensor(at::kFloat).tensor({gate_size})),
                       "bias_hh_l" + std::to_string(i)));
  }
  this->flatten_parameters();
}

template <typename Derived>
void RNNBase<Derived>::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(hidden_size_);
  for (auto& p : this->parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

template <typename Derived>
variable_list RNNBase<Derived>::GRU_cell_forward(variable_list inputs, int i) {
  auto x = inputs[0];
  auto hx = inputs[1].defined()
      ? inputs[1]
      : Var(this->DefaultTensor(at::kFloat).zeros({x.size(0), hidden_size_}));

  auto gi = linear(x, ihw[i], ihb[i]);
  auto gh = linear(x, hhw[i], hhb[i]);
  auto gic = gi.chunk(3, 1);
  auto ghc = gh.chunk(3, 1);

  auto reset_gate = (gic[0] + ghc[0]).sigmoid_();
  auto input_gate = (gic[1] + ghc[1]).sigmoid_();
  auto new_gate = (gic[2] + reset_gate * ghc[2]).tanh_();
  auto hy = new_gate + input_gate * (hx - new_gate);

  return variable_list({hy});
}

template <typename Derived>
variable_list RNNBase<Derived>::RNN_TANH_cell_forward(
    variable_list inputs,
    int i) {
  auto x = inputs[0];
  auto hx = inputs[1].defined()
      ? inputs[1]
      : Var(this->DefaultTensor(at::kFloat).zeros({x.size(0), hidden_size_}));

  auto h = (linear(x, ihw[i], ihb[i]) + linear(hx, hhw[i], hhb[i])).tanh();
  return variable_list({h});
}

template <typename Derived>
variable_list RNNBase<Derived>::RNN_RELU_cell_forward(
    variable_list inputs,
    int i) {
  auto x = inputs[0];
  auto hx = inputs[1].defined()
      ? inputs[1]
      : Var(this->DefaultTensor(at::kFloat).zeros({x.size(0), hidden_size_}));

  auto h = at::relu(linear(x, ihw[i], ihb[i]) + linear(hx, hhw[i], hhb[i]));
  return variable_list({h});
}

template <typename Derived>
variable_list RNNBase<Derived>::LSTM_cell_forward(variable_list inputs, int i) {
  auto x = inputs[0];
  auto hid = inputs[1].defined()
      ? inputs[1]
      : Var(this->DefaultTensor(at::kFloat)
                .zeros({2, x.size(0), hidden_size_}));
  auto hx = hid[0];
  auto cx = hid[1];

  auto gates = linear(x, ihw[i], ihb[i]) + linear(hx, hhw[i], hhb[i]);

  auto chunked = gates.chunk(4, 1);
  auto in_gate = chunked[0].sigmoid();
  auto forget_gate = chunked[1].sigmoid();
  auto cell_gate = chunked[2].tanh();
  auto out_gate = chunked[3].sigmoid();

  auto cy = (forget_gate * cx) + (in_gate * cell_gate);
  auto hy = out_gate * cy.tanh();

  return variable_list({at::stack({hy, cy}, 0)});
}

template <typename Derived>
variable_list RNNBase<Derived>::cell_forward(variable_list inputs, int i) {
  if (mode_ == RNNMode::LSTM)
    return LSTM_cell_forward(inputs, i);
  else if (mode_ == RNNMode::GRU)
    return GRU_cell_forward(inputs, i);
  else if (mode_ == RNNMode::RNN_TANH)
    return RNN_TANH_cell_forward(inputs, i);
  else if (mode_ == RNNMode::RNN_RELU)
    return RNN_RELU_cell_forward(inputs, i);
  else
    throw std::runtime_error("No such RNN mode");
}

template <typename Derived>
variable_list RNNBase<Derived>::autograd_forward(variable_list inputs) {
  auto inp = inputs[0];

  std::vector<Tensor> hidden;
  auto hasHidden = inputs[1].defined();
  auto layerDim = hasHidden ? inputs[1].ndimension() - 3 : -1;
  for (size_t i = 0; i < nlayers_; i++) {
    hidden.push_back(hasHidden ? inputs[1].select(layerDim, i) : Variable());
  }

  auto output =
      Var(this->DefaultTensor(at::kFloat)
              .zeros({inp.size(0), inp.size(1), hidden_size_}),
          false);
  for (auto t = 0U; t < inp.size(0); t++) {
    auto x = inp.select(0, t);
    for (size_t i = 0; i < nlayers_; i++) {
      auto layer_output = cell_forward({x, hidden[i]}, i);
      hidden[i] = layer_output[0];
      if (mode_ == RNNMode::LSTM) {
        x = hidden[i][0];
      } else {
        x = hidden[i];
      }
      auto output_slice = output.select(0, t);
      output_slice.copy_(x);
      if (dropout_ > 0 && i != nlayers_ - 1) {
        x = dropout_module->forward({x})[0];
      }
    }
  }

  auto hidout = at::stack(hidden, 0);
  if (mode_ == RNNMode::LSTM) {
    hidout.transpose_(0, 1);
  }
  return variable_list({output, hidout});
}

template <typename Derived>
bool RNNBase<Derived>::flatten_parameters() {
  data_ptrs_.clear();
  auto anyParam = ihw[0];
  if (!anyParam.is_cuda() || !at::cudnn_is_acceptable(anyParam)) {
    return false;
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
    return false;
  }

  std::vector<Tensor> weight_list;
  for (size_t i = 0; i < nlayers_; i++) {
    weight_list.push_back(ihw[i]);
    weight_list.push_back(hhw[i]);
    if (!no_bias_) {
      weight_list.push_back(ihb[i]);
      weight_list.push_back(hhb[i]);
    }
  }
  auto weight_stride0 = no_bias_ ? 2 : 4;

  {
    no_grad_guard guard;
    flat_weight_ = at::_cudnn_rnn_flatten_weight(
        weight_list,
        weight_stride0,
        input_size_,
        mode_,
        hidden_size_,
        nlayers_,
        false,
        false); // batch_first and bidirectional, unsupported
  }
  for (auto& p : params) {
    data_ptrs_.emplace_back(p.second.data().data_ptr());
  }
  return true;
}

template <typename Derived>
variable_list RNNBase<Derived>::CUDNN_forward(variable_list inputs) {
  std::vector<Tensor> weight_list;
  for (size_t i = 0; i < nlayers_; i++) {
    weight_list.push_back(ihw[i]);
    weight_list.push_back(hhw[i]);
    if (!no_bias_) {
      weight_list.push_back(ihb[i]);
      weight_list.push_back(hhb[i]);
    }
  }
  auto weight_stride0 = no_bias_ ? 2 : 4;

  auto x = inputs[0];
  Variable hx, cx;
  if (!inputs[1].defined()) {
    hx = x.type().zeros({nlayers_, x.size(1), hidden_size_});
    if (mode_ == RNNMode::LSTM) {
      cx = x.type().zeros({nlayers_, x.size(1), hidden_size_});
    }
  } else {
    hx = mode_ == RNNMode::LSTM ? inputs[1][0] : inputs[1];
    cx = mode_ == RNNMode::LSTM ? inputs[1][1] : Variable();
  }
  auto dropout_state = x.type().tensor();

  std::vector<void*> weight_data_ptrs;
  auto params = this->parameters();
  for (auto& p : params) {
    weight_data_ptrs.emplace_back(p.second.data().data_ptr());
  }
  if (weight_data_ptrs != data_ptrs_) {
    std::cerr << "Parameters are unflattened! Code path might be super slow. "
                 "Please call flatten_parameters() when you muck around with "
                 "storages!"
              << std::endl;
    flat_weight_ = Variable();
  }

  // tup = std::tuple of output, hy, cy, reserve, new_weight_buf
  auto tup = _cudnn_rnn(
      x,
      weight_list,
      weight_stride0,
      flat_weight_,
      hx,
      cx,
      mode_,
      hidden_size_,
      nlayers_,
      false, // batch first
      0, // TODO Use C++ dropout descriptors
      this->train_,
      false, // bidirectional
      {}, // packing not supported
      dropout_state // TODO waiting on dropout state descriptor in C++ pytorch
  );

  Variable hidout = mode_ == RNNMode::LSTM
      ? at::stack({std::get<1>(tup), std::get<2>(tup)}, 0)
      : std::get<1>(tup);
  Variable output = std::get<0>(tup);
  return variable_list({output, hidout});
}

template <typename Derived>
variable_list RNNBase<Derived>::forward(variable_list inputs) {
  variable_list inp;
  inp.push_back(inputs[0]);
  if (inputs.size() > 1) {
    inp.push_back(inputs[1]);
  } else {
    inp.push_back(Variable());
  }

  // TODO implement with dropout descriptors
  auto output = at::cudnn_is_acceptable(inp[0]) && dropout_ == 0
      ? CUDNN_forward(inp)
      : autograd_forward(inp);

  return output;
}

template <typename Derived>
void RNNBase<Derived>::cuda() {
  nn::Module::cuda();
  flatten_parameters();
}

template <typename Derived>
void RNNBase<Derived>::cpu() {
  nn::Module::cpu();
  flatten_parameters();
}

template class RNNBase<LSTM>;
template class RNNBase<GRU>;
template class RNNBase<RNN>;
}} // namespace torch::nn
