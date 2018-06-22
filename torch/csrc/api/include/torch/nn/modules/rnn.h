#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace nn {
namespace detail {
struct RNNOptionsBase {
  RNNOptionsBase(int64_t input_size, int64_t hidden_size);
  virtual ~RNNOptionsBase() = default;
  TORCH_ARG(int64_t, input_size);
  TORCH_ARG(int64_t, hidden_size);
  TORCH_ARG(int64_t, layers) = 1;
  TORCH_ARG(bool, with_bias) = true;
  TORCH_ARG(double, dropout) = 0.0;
};

template <typename Derived>
class RNNImplBase : public torch::nn::Cloneable<Derived> {
 public:
  // These must line up with the CUDNN mode codes:
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
  enum class CuDNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

  RNNImplBase(
      RNNOptionsBase options,
      at::optional<CuDNNMode> cudnn_mode = at::nullopt,
      int64_t number_of_gates = 1,
      bool has_cell_state = false);

  std::vector<Variable> forward(std::vector<Variable>);

  void reset() override;

  void to(at::Type& type) override;
  void to(at::ScalarType scalar_type) override;
  void to(at::Backend backend) override;

 protected:
  virtual std::vector<Variable> cell_forward(
      std::vector<Variable>,
      int64_t layer) = 0;

  std::vector<Variable> CUDNN_forward(std::vector<Variable>);
  std::vector<Variable> autograd_forward(std::vector<Variable>);

  void flatten_parameters_for_cudnn();
  std::vector<at::Tensor> flat_weights() const;

  RNNOptionsBase options_;

  std::vector<Variable> ihw_;
  std::vector<Variable> ihb_;
  std::vector<Variable> hhw_;
  std::vector<Variable> hhb_;

  int64_t number_of_gates_;
  bool has_cell_state_;
  at::optional<CuDNNMode> cudnn_mode_;
  Dropout dropout_module_;

  // This is copied from pytorch, to determine whether weights are flat for the
  // fast CUDNN route. Otherwise, we have to use non flattened weights, which
  // are much slower.
  // https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/torch/nn/modules/rnn.py#L159-L165
  // TODO Actually since we are in C++ we can probably just actually check if
  // the parameters are flat, instead of relying on data pointers and stuff.
  std::vector<void*> data_ptrs_;
  Variable flat_weights_;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// TODO: Replace this with passing an activation module.

enum class RNNActivation { ReLU, Tanh };

struct RNNOptions {
  RNNOptions(int64_t input_size, int64_t hidden_size);

  RNNOptions& tanh();
  RNNOptions& relu();

  TORCH_ARG(int64_t, input_size);
  TORCH_ARG(int64_t, hidden_size);
  TORCH_ARG(int64_t, layers) = 1;
  TORCH_ARG(bool, with_bias) = true;
  TORCH_ARG(double, dropout) = 0.0;
  TORCH_ARG(RNNActivation, activation) = RNNActivation::ReLU;
};

class RNNImpl : public detail::RNNImplBase<RNNImpl> {
 public:
  explicit RNNImpl(RNNOptions options);

  const RNNOptions& options() const noexcept;

 private:
  std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer)
      override;

  RNNOptions options_;
  std::function<Variable(Variable)> activation_function_;
};

TORCH_MODULE(RNN);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using LSTMOptions = detail::RNNOptionsBase;

class LSTMImpl : public detail::RNNImplBase<LSTMImpl> {
 public:
  explicit LSTMImpl(LSTMOptions options);

  const LSTMOptions& options() const noexcept;

 private:
  std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer)
      override;
};

TORCH_MODULE(LSTM);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using GRUOptions = detail::RNNOptionsBase;

class GRUImpl : public detail::RNNImplBase<GRUImpl> {
 public:
  explicit GRUImpl(GRUOptions options);

  const GRUOptions& options() const noexcept;

 private:
  std::vector<Variable> cell_forward(std::vector<Variable>, int64_t layer)
      override;
};

TORCH_MODULE(GRU);

} // namespace nn
} // namespace torch
