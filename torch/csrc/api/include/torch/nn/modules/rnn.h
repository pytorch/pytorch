#pragma once

#include <torch/detail.h>
#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/Error.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace torch { namespace nn {
template <typename Derived>
class RNNBase : public CloneableModule<Derived> {
 public:
  // These must line up with the CUDNN mode codes
  enum RNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

  RNNBase(
      uint32_t input_size,
      uint32_t hidden_size,
      int mode,
      uint32_t nlayers,
      bool with_bias,
      float dropout);

  using CloneableModule<Derived>::parameters;
  using CloneableModule<Derived>::is_training;

  bool flatten_parameters(); // Flatten for cudnn

  variable_list forward(variable_list) override;

  void cpu() override;
  void cuda() override;

  std::vector<Variable> ihw;
  std::vector<Variable> ihb;
  std::vector<Variable> hhw;
  std::vector<Variable> hhb;

 private:
  using CloneableModule<Derived>::add;

  uint32_t input_size_;
  uint32_t hidden_size_;
  uint32_t gate_size_;
  RNNMode mode_;
  uint32_t nlayers_;
  bool with_bias_;
  float dropout_;

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
  LSTM(
      uint32_t input_size,
      uint32_t hidden_size,
      uint32_t nlayers = 1,
      bool with_bias = true,
      float dropout = 0)
      : RNNBase(
            input_size,
            hidden_size,
            RNNMode::LSTM,
            nlayers,
            with_bias,
            dropout) {}
};

class GRU : public RNNBase<GRU> {
 public:
  GRU(uint32_t input_size,
      uint32_t hidden_size,
      uint32_t nlayers = 1,
      bool with_bias = true,
      float dropout = 0)
      : RNNBase(
            input_size,
            hidden_size,
            RNNMode::GRU,
            nlayers,
            with_bias,
            dropout) {}
};

class RNN : public RNNBase<RNN> {
 public:
  enum Mode { Tanh = RNNMode::RNN_TANH, Relu = RNNMode::RNN_RELU };
  RNN(uint32_t input_size,
      uint32_t hidden_size,
      Mode mode = Mode::Tanh,
      uint32_t nlayers = 1,
      bool with_bias = true,
      float dropout = 0)
      : RNNBase(input_size, hidden_size, mode, nlayers, with_bias, dropout) {}
};
}} // namespace torch::nn
