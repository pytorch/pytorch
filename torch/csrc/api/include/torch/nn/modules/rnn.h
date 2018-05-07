#pragma once

#include <torch/detail.h>
#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace torch { namespace nn {
template <typename Derived>
class RNNBase : public nn::CloneableModule<Derived> {
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
  std::shared_ptr<nn::Module> dropout_module;

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
}} // namespace torch::nn
