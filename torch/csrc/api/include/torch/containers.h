#pragma once

#include "detail.h"

#include "torch/csrc/autograd/variable.h"

#define AUTOGRAD_CONTAINER_CLASS(Type) \
  class Type : public torch::Container_CRTP<Type>

namespace torch {
class ContainerImpl {
 public:
  virtual ~ContainerImpl() = default;
  // Only construct parameters in initialize_parameters, and
  // containers in initialize_containers. Most of the time, the containers are
  // the only thing you need to add.
  // You are guaranteed that containers are added before parameters.
  virtual void initialize_containers(){};
  virtual void initialize_parameters(){};
  virtual void reset_parameters(){};

  virtual variable_list forward(variable_list) = 0;
  virtual Container clone() const = 0;

  std::map<std::string, Variable> parameters() const;
  Variable& param(std::string const&);

  virtual void cuda();
  virtual void cpu();
  void train();
  void eval();

  at::Type& DefaultTensor(at::ScalarType s);

  std::unordered_map<std::string, Container> children_;
  std::unordered_map<std::string, Variable> params_;
  bool cuda_ = false;
  bool train_ = true;

  template <class Archive>
  void save(Archive& ar) const {
    auto params = parameters();
    std::size_t size = params.size();
    ar(size);
    for (auto& p : params) {
      ar(p.first, p.second);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    auto params = parameters();
    std::size_t size;
    ar(size);
    std::string name;
    for (std::size_t i = 0; i < size; i++) {
      ar(name);
      ar(params[name]);
    }
  }

 protected:
  Container add(Container, std::string const&);
  // Be careful when registering Tensors that are not variables
  Variable& add(Variable, std::string const&);
};

template <class Derived>
class Container_CRTP : public ContainerImpl {
 public:
  std::shared_ptr<Derived> make() const {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->initialize_containers();
    ptr->initialize_parameters();
    ptr->reset_parameters();
    return ptr;
  }

  Container clone() const override {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->children_.clear();
    ptr->params_.clear();
    ptr->initialize_containers();
    ptr->initialize_parameters();
    auto newParams = ptr->parameters();
    for (auto& param : parameters()) {
      newParams[param.first].data().copy_(param.second.data());
    }
    if (cuda_) {
      ptr->cuda();
    } else {
      ptr->cpu();
    }
    return ptr;
  }
};

template <class Derived>
class ContainerListImpl : public Container_CRTP<Derived> {
  // Lets you use a container like a vector without making a new class,
  // just for simple implementations
 public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error(
        "ContainerList has no forward, maybe you"
        " wanted to subclass and override this function?");
  }

  Container add(Container m) {
    return append(m).children_.back();
  }

  ContainerListImpl<Derived>& append(Container m) {
    children_.push_back(m);
    ContainerImpl::add(children_.back(), std::to_string(size() - 1));
    return *this;
  }

  Container& operator[](int index) {
    return children_[index];
  }

  int size() {
    return children_.size();
  }

  std::vector<Container>::iterator begin() {
    return children_.begin();
  }

  std::vector<Container>::iterator end() {
    return children_.end();
  }

  std::vector<Container> children_;
};

class ContainerList : public ContainerListImpl<ContainerList> {};

class Sequential : public ContainerListImpl<Sequential> {
  // Mimics nn.Sequential from pytorch.
 public:
  variable_list forward(variable_list input) override {
    for (auto& container : children_) {
      input = container->forward(input);
    }
    return input;
  }

  Container add(Container m, std::string name = "") {
    return append(m, name).children_.back();
  }

  Sequential& append(Container m, std::string name = "") {
    if (name == "") {
      name = std::to_string(size());
    }
    children_.push_back(m);
    ContainerImpl::add(children_.back(), name);
    return *this;
  }
};

AUTOGRAD_CONTAINER_CLASS(SimpleContainer) {
  // Lets you use a container without making a new class,
  // for experimental implementations
 public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error(
        "SimpleContainer has no forward, maybe you"
        " wanted to subclass and override this function?");
  }
  using ContainerImpl::add;
};

AUTOGRAD_CONTAINER_CLASS(Functional) {
  // Lets you create a container from a function, designed for use in
  // Sequential.
 public:
  Functional(std::function<variable_list(variable_list)> fun) : fun_(fun){};
  Functional(std::function<Variable(Variable)> fun)
      : fun_([fun](variable_list input) {
          return variable_list({fun(input[0])});
        }){};

  variable_list forward(variable_list input) override {
    return fun_(input);
  };

  std::function<variable_list(variable_list)> fun_;
};

AUTOGRAD_CONTAINER_CLASS(Linear) {
 public:
  Linear(uint32_t nin, uint32_t nout) : nin(nin), nout(nout) {}

  variable_list forward(variable_list) override;
  void reset_parameters() override;
  void initialize_parameters() override;
  AUTOGRAD_KWARG(Linear, bool, no_bias, false, true);

  Variable weight, bias;
  uint32_t nin, nout;
};

AUTOGRAD_CONTAINER_CLASS(Embedding) {
 public:
  Embedding(uint32_t num_embeddings, uint32_t embedding_dim)
      : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {}

  variable_list forward(variable_list) override;
  void reset_parameters() override;
  void initialize_parameters() override;

  Variable weight;
  uint32_t num_embeddings, embedding_dim;
};

AUTOGRAD_CONTAINER_CLASS(Conv) {
 private:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan)
      : Nd_(Nd),
        in_channels_(in_chan),
        out_channels_(out_chan),
        stride_(makeTup(1, 1)),
        padding_(makeTup(0)),
        dilation_(makeTup(1, 1)),
        dilated_(false),
        output_padding_(makeTup(0)) {}

 public:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan, int ks)
      : Conv(Nd, in_chan, out_chan) {
    ks_ = makeTup(ks, 1);
  }

  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan, IntVec ks)
      : Conv(Nd, in_chan, out_chan) {
    ks_ = makeTup(ks);
  }

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  template <typename T>
  Conv& stride(T s) {
    stride_ = makeTup(s, 1);
    return *this;
  }
  template <typename T>
  Conv& padding(T s) {
    padding_ = makeTup(s);
    return *this;
  }
  template <typename T>
  Conv& dilation(T s) {
    dilation_ = makeTup(s, 1);
    return *this;
  }
  template <typename T>
  Conv& output_padding(T s) {
    output_padding_ = makeTup(s);
    return *this;
  }

  AUTOGRAD_KWARG(Conv, bool, transposed, false, true)
  AUTOGRAD_KWARG(Conv, bool, no_bias, false, true)
  AUTOGRAD_KWARG(Conv, int, groups, 1, 1)

  Variable weight, bias;
  uint32_t Nd_;
  uint32_t in_channels_;
  uint32_t out_channels_;
  IntVec ks_;
  IntVec stride_;
  IntVec padding_;
  IntVec dilation_;
  bool dilated_;
  IntVec output_padding_;

 protected:
  IntVec makeTup(int x, int def = 0) {
    IntVec ret;
    if (Nd_ == 1) {
      ret.push_back(x);
      ret.push_back(def);
    } else {
      for (auto i = 0U; i < Nd_; i++)
        ret.push_back(x);
    }
    return ret;
  }
  IntVec makeTup(IntVec x) {
    return x;
  }
};

class Conv1d : public Conv {
 public:
  Conv1d(uint32_t i, uint32_t o, int ks) : Conv(1, i, o, ks) {}
  Conv1d(uint32_t i, uint32_t o, IntVec ks) : Conv(1, i, o, ks) {}
};

class Conv2d : public Conv {
 public:
  Conv2d(uint32_t i, uint32_t o, int ks) : Conv(2, i, o, ks) {}
  Conv2d(uint32_t i, uint32_t o, IntVec ks) : Conv(2, i, o, ks) {}
};

class Conv3d : public Conv {
 public:
  Conv3d(uint32_t i, uint32_t o, int ks) : Conv(3, i, o, ks) {}
  Conv3d(uint32_t i, uint32_t o, IntVec ks) : Conv(3, i, o, ks) {}
};

AUTOGRAD_CONTAINER_CLASS(BatchNorm) {
 public:
  BatchNorm(uint32_t num_features) : num_features_(num_features) {}

  AUTOGRAD_KWARG(BatchNorm, double, eps, 1e-5, 1e-5)
  AUTOGRAD_KWARG(BatchNorm, double, momentum, 0.1, 0.1)
  AUTOGRAD_KWARG(BatchNorm, bool, affine, true, true)
  AUTOGRAD_KWARG(BatchNorm, bool, stateful, false, true)

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  Variable weight;
  Variable bias;
  Variable running_mean;
  Variable running_var;

 protected:
  uint32_t num_features_;
};

AUTOGRAD_CONTAINER_CLASS(Dropout) {
 public:
  Dropout(double p = 0.5) : p_(p) {
    assert(p < 1 && p >= 0);
  }
  variable_list forward(variable_list) override;

 protected:
  double p_;
};

AUTOGRAD_CONTAINER_CLASS(Dropout2d) {
 public:
  Dropout2d(double p = 0.5) : p_(p) {
    assert(p < 1 && p >= 0);
  }
  variable_list forward(variable_list) override;

 protected:
  double p_;
};

template <typename Derived>
class RNNBase : public Container_CRTP<Derived> {
 public:
  // These must line up with the CUDNN mode codes
  enum RNNMode : int64_t { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };
  RNNBase(uint32_t input_size, uint32_t hidden_size)
      : input_size_(input_size), hidden_size_(hidden_size) {}

  AUTOGRAD_KWARG(RNNBase, RNNMode, mode, RNNMode::LSTM, RNNMode::LSTM)
  AUTOGRAD_KWARG(RNNBase, uint32_t, nlayers, 1, 1);
  AUTOGRAD_KWARG(RNNBase, bool, no_bias, false, true)
  AUTOGRAD_KWARG(RNNBase, float, dropout, 0, 0)

  bool flatten_parameters(); // Flatten for cudnn

  variable_list forward(variable_list) override;
  void initialize_containers() override;
  void initialize_parameters() override;
  void reset_parameters() override;

  void cpu() override;
  void cuda() override;

  std::vector<Variable> ihw;
  std::vector<Variable> ihb;
  std::vector<Variable> hhw;
  std::vector<Variable> hhb;

 protected:
  uint32_t input_size_;
  uint32_t hidden_size_;
  uint32_t gate_size_;
  // This is copied from pytorch, to determine whether weights are flat for
  // the fast CUDNN route. Otherwise, we have to use non flattened weights,
  // which
  // are much slower.
  // https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/torch/nn/modules/rnn.py#L159-L165
  // TODO Actually since we are in C++ we can probably just actually check if
  // the parameters are flat, instead of relying on data pointers and stuff.
  std::vector<void*> data_ptrs_;
  Variable flat_weight_;
  Container dropout_module;

  variable_list CUDNN_forward(variable_list);
  variable_list autograd_forward(variable_list);

  variable_list cell_forward(variable_list, int);
  variable_list LSTM_cell_forward(variable_list, int);
  variable_list GRU_cell_forward(variable_list, int);
  variable_list RNN_RELU_cell_forward(variable_list, int);
  variable_list RNN_TANH_cell_forward(variable_list, int);
};

// We must instantiate these templates so we can put implementations in the .cpp
class LSTM : public RNNBase<LSTM> {
 public:
  LSTM(uint32_t inp_size, uint32_t hid_size) : RNNBase(inp_size, hid_size) {
    mode_ = RNNBase::RNNMode::LSTM;
  }
};

class GRU : public RNNBase<GRU> {
 public:
  GRU(uint32_t inp_size, uint32_t hid_size) : RNNBase(inp_size, hid_size) {
    mode_ = RNNBase::RNNMode::GRU;
  }
};

class RNN : public RNNBase<RNN> {
 public:
  enum Mode { Tanh, Relu };
  RNN(uint32_t inp_size, uint32_t hid_size, Mode mode = Mode::Tanh)
      : RNNBase(inp_size, hid_size) {
    if (mode == Mode::Tanh) {
      mode_ = RNNBase::RNNMode::RNN_TANH;
    } else if (mode == Mode::Relu) {
      mode_ = RNNBase::RNNMode::RNN_RELU;
    } else {
      throw std::runtime_error("RNN Mode not supported");
    }
  }
};

} // namespace torch
