#include "torch/containers.h"

namespace torch {
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
}

std::map<std::string, Variable> ContainerImpl::parameters() const {
  std::map<std::string, Variable> ret;
  for (auto pair : children_) {
    auto& name = pair.first;
    auto& child = pair.second;
    for (auto& p : child->parameters()) {
      ret[name + "." + p.first] = p.second;
    }
  }
  for (auto pair : params_) {
    ret[pair.first] = pair.second;
  }
  return ret;
}

Variable& ContainerImpl::param(std::string const& name) {
  ContainerImpl* container = this;
  auto begin = 0;
  while (true) {
    auto dot_pos = name.find('.', begin);
    if (dot_pos == std::string::npos) {
      break;
    }

    auto child_name = name.substr(begin, dot_pos - begin);
    auto it = container->children_.find(child_name);
    if (it == container->children_.end()) {
      throw std::runtime_error("No such child: " + child_name);
    }

    container = it->second.get();
    begin = dot_pos + 1; // Skip the dot
  }

  auto param_name = name.substr(begin);
  auto it = container->params_.find(param_name);
  if (it == params_.end()) {
    throw std::runtime_error("No such param: " + param_name);
  }
  return it->second;
}

void ContainerImpl::cuda() {
  for (auto& pair : children_) {
    pair.second->cuda();
  }
  cuda_ = true;
  auto copied = params_;
  params_.clear();
  initialize_parameters();
  for (auto pair : params_) {
    pair.second.data().copy_(copied[pair.first].data());
  }
}

void ContainerImpl::cpu() {
  for (auto& pair : children_) {
    pair.second->cpu();
  }
  cuda_ = false;
  auto copied = params_;
  params_.clear();
  initialize_parameters();
  for (auto pair : params_) {
    pair.second.data().copy_(copied[pair.first].data());
  }
}

void ContainerImpl::train() {
  for (auto& pair : children_) {
    pair.second->train();
  }
  train_ = true;
}

void ContainerImpl::eval() {
  for (auto& pair : children_) {
    pair.second->eval();
  }
  train_ = false;
}

Container ContainerImpl::add(Container m, std::string const& name) {
  if (this->children_.find(name) != this->children_.end()) {
    throw std::runtime_error("Trying to add container that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow containers with dots in their names, as that would make
    // their parameters not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->children_[name] = std::move(m);
  return this->children_[name];
}

Variable& ContainerImpl::add(Variable v, std::string const& name) {
  if (this->params_.find(name) != this->params_.end()) {
    throw std::runtime_error("Trying to add parameter that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow parameters with dots in their names, as that would make
    // them not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->params_[name] = v;
  return this->params_[name];
}

at::Type& ContainerImpl::DefaultTensor(at::ScalarType s) {
  if (cuda_)
    return at::CUDA(s);
  else
    return at::CPU(s);
}

variable_list Linear::forward(variable_list input) {
  return variable_list({linear(input[0], weight, bias)});
}

void Linear::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(weight.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

void Linear::initialize_parameters() {
  weight = this->add(
      Var(DefaultTensor(at::kFloat).tensor({nout, nin}), true), "weight");
  if (!no_bias_) {
    bias =
        this->add(Var(DefaultTensor(at::kFloat).tensor({nout}), true), "bias");
  }
}

variable_list Embedding::forward(variable_list input) {
  auto x = input[0];
  return variable_list({at::embedding(weight, x, -1, false, false)});
}

void Embedding::reset_parameters() {
  for (auto& p : parameters()) {
    p.second.data().normal_(0, 1);
  }
}

void Embedding::initialize_parameters() {
  weight = this->add(
      Var(DefaultTensor(at::kFloat).tensor({num_embeddings, embedding_dim}),
          true),
      "weight");
}

void Conv::initialize_parameters() {
  if (!transposed_) {
    for (auto pad : output_padding_) {
      if (pad != 0) {
        throw std::runtime_error(
            "Only transposed convolutions support output padding!");
      }
    }
  }

  IntVec wsize;
  if (transposed_) {
    wsize.push_back(in_channels_);
    wsize.push_back(out_channels_ / groups_);
  } else {
    wsize.push_back(out_channels_);
    wsize.push_back(in_channels_ / groups_);
  }
  wsize.insert(wsize.end(), ks_.begin(), ks_.end());
  weight =
      this->add(Var(DefaultTensor(at::kFloat).tensor(wsize), true), "weight");
  if (!no_bias_) {
    bias = this->add(
        Var(DefaultTensor(at::kFloat).tensor({out_channels_}), true), "bias");
  } else {
    assert(!bias.defined());
  }
}

void Conv::reset_parameters() {
  auto n = in_channels_;
  for (auto k : ks_)
    n *= k;
  auto stdv = 1.0 / std::sqrt(n);
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list Conv::forward(variable_list input) {
  auto x = input[0];
  if (Nd_ == 1) {
    assert(x.ndimension() == 3);
    x = x.unsqueeze(-1); // TODO: Use conv1d once available
  } else if (Nd_ == 2) {
    assert(x.ndimension() == 4);
  } else if (Nd_ == 3) {
    assert(x.ndimension() == 5);
  } else {
    throw std::runtime_error("Only Conv{1,2,3}d are supported");
  }

  Variable out;
  if (Nd_ == 1 || Nd_ == 2) {
    if (transposed_) {
      out = at::conv_transpose2d(
          x,
          weight,
          bias,
          stride_,
          padding_,
          output_padding_,
          groups_,
          dilation_);
    } else {
      out = at::conv2d(x, weight, bias, stride_, padding_, dilation_, groups_);
    }
  } else if (Nd_ == 3) {
    if (transposed_) {
      out = at::conv_transpose3d(
          x,
          weight,
          bias,
          stride_,
          padding_,
          output_padding_,
          groups_,
          dilation_);
    } else {
      out = at::conv3d(x, weight, bias, stride_, padding_, dilation_, groups_);
    }
  }

  return variable_list({out});
}

void BatchNorm::initialize_parameters() {
  if (affine_) {
    weight = this->add(
        Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "weight");
    bias = this->add(
        Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "bias");
  }

  if (stateful_) {
    running_mean = Var(DefaultTensor(at::kFloat).zeros({num_features_}), false);
    running_var = Var(DefaultTensor(at::kFloat).ones({num_features_}), false);
  }
}

void BatchNorm::reset_parameters() {
  if (affine_) {
    weight.data().uniform_();
    bias.data().zero_();
  }

  if (stateful_) {
    running_mean.data().zero_();
    running_var.data().fill_(1);
  }
}

variable_list BatchNorm::forward(variable_list inputs) {
  auto& input = inputs[0];
  auto& running_mean = (stateful_ ? this->running_mean : inputs[1]);
  auto& running_var = (stateful_ ? this->running_var : inputs[2]);

  if (train_) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    if (input.numel() / num_channels <= 1) {
      throw std::runtime_error(
          "BatchNorm expected more than 1 value per channel when training!");
    }
  }

  auto output = at::batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      train_,
      momentum_,
      eps_,
      hasCudnn());

  return variable_list({output});
}

template <typename Derived>
void RNNBase<Derived>::initialize_containers() {
  if (dropout_ > 0)
    dropout_module = Dropout(dropout_).make();
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
    ihb.push_back(no_bias_ ? Variable() : this->add(
          Var(this->DefaultTensor(at::kFloat).tensor({gate_size})),
          "bias_ih_l" + std::to_string(i)));
    hhb.push_back(no_bias_ ? Variable() : this->add(
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
  Container_CRTP<Derived>::cuda();
  flatten_parameters();
}

template <typename Derived>
void RNNBase<Derived>::cpu() {
  Container_CRTP<Derived>::cpu();
  flatten_parameters();
}

template class RNNBase<LSTM>;
template class RNNBase<GRU>;
template class RNNBase<RNN>;

variable_list Dropout::forward(variable_list inputs) {
  if (p_ == 0 || !this->train_)
    return inputs;
  variable_list lst;
  for (auto x : inputs) {
    auto noise = x.data().type().tensor(x.sizes());
    noise = (noise.uniform_(0, 1) > p_)
                .toType(x.type().scalarType())
                .mul_(1. / (1 - p_));
    lst.push_back(x * Var(noise));
  }
  return lst;
}

variable_list Dropout2d::forward(variable_list inputs) {
  if (p_ == 0 || !this->train_)
    return inputs;
  variable_list lst;
  for (auto x : inputs) {
    auto noise = x.data().type().tensor({x.size(0), x.size(1), 1, 1});
    noise = (noise.uniform_(0, 1) > p_)
                .toType(x.type().scalarType())
                .mul_(1. / (1 - p_));
    lst.push_back(x * Var(noise));
  }
  return lst;
}

} // namespace torch
