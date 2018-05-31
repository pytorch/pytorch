#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace nn {
class Dropout;
}
} // namespace torch

namespace torch {
namespace nn {

template <typename Derived>
class RNNBase : public CloneableModule<Derived> {
 public:
  // These must line up with the CUDNN mode codes:
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
  enum class CuDNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

  RNNBase(
      int64_t input_size,
      int64_t hidden_size,
      at::optional<CuDNNMode> cudnn_mode = at::nullopt,
      int64_t number_of_gates = 1,
      bool has_cell_state = false);

  void reset() override;

  std::vector<Variable> forward(std::vector<Variable>);

  void to(at::Type& type) override;
  void to(at::ScalarType scalar_type) override;
  void to(at::Backend backend) override;

  TORCH_ATTR(int64_t, input_size);
  TORCH_ATTR(int64_t, hidden_size);
  TORCH_ATTR(int64_t, layers) = 1;
  TORCH_ATTR(bool, with_bias) = true;
  TORCH_ATTR(double, dropout) = 0.0;

 protected:
  virtual std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer) = 0;

  std::vector<Variable> CUDNN_forward(std::vector<Variable>);
  std::vector<Variable> autograd_forward(std::vector<Variable>);

  void flatten_parameters_for_cudnn();
  std::vector<at::Tensor> flat_weights() const;

  std::vector<Variable> ihw_;
  std::vector<Variable> ihb_;
  std::vector<Variable> hhw_;
  std::vector<Variable> hhb_;

  int64_t number_of_gates_;
  bool has_cell_state_;
  at::optional<CuDNNMode> cudnn_mode_;
  std::shared_ptr<Dropout> dropout_module_;

  // This is copied from pytorch, to determine whether weights are flat for the
  // fast CUDNN route. Otherwise, we have to use non flattened weights, which
  // are much slower.
  // https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/torch/nn/modules/rnn.py#L159-L165
  // TODO Actually since we are in C++ we can probably just actually check if
  // the parameters are flat, instead of relying on data pointers and stuff.
  std::vector<void*> data_ptrs_;
  Variable flat_weights_;
};

class LSTM : public RNNBase<LSTM> {
 public:
  LSTM(int64_t input_size, int64_t hidden_size);

 private:
  std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer) override;
};

class GRU : public RNNBase<GRU> {
 public:
  GRU(int64_t input_size, int64_t hidden_size);

 private:
  std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer) override;
};

class RNN : public RNNBase<RNN> {
 public:
  enum class Activation { ReLU, Tanh };

  RNN(int64_t input_size, int64_t hidden_size);

  void reset() override;

  RNN& relu();
  RNN& tanh();

  TORCH_ATTR(Activation, activation) = Activation::Tanh;

 private:
  using ActivationFunction = std::function<Variable(Variable)>;

  std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer) override;

  ActivationFunction activation_function_;
};
} // namespace nn
} // namespace torch
