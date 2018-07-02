#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>
#include <ATen/Error.h>
#include <ATen/optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace nn {

struct RNNOutput {
  Tensor output;
  Tensor state;
};

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
      RNNOptionsBase options_,
      at::optional<CuDNNMode> cudnn_mode = at::nullopt,
      int64_t number_of_gates = 1,
      bool has_cell_state = false);

  RNNOutput forward(Tensor input, Tensor state = {});

  void reset() override;

  /// Recursively casts all parameters to the given device and dtype.
  void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false)
      override;

  /// Recursively casts all parameters to the given dtype.
  void to(torch::Dtype dtype, bool non_blocking = false) override;

  /// Recursively moves all parameters to the given device.
  void to(torch::Device device, bool non_blocking = false) override;

  /// Fills the internal flattened parameter buffers passed to cuDNN. Call this
  /// method if you mess around with the variable storages and want to use
  /// cuDNN.
  void flatten_parameters_for_cudnn();

  RNNOptionsBase options;

  std::vector<Tensor> w_ih;
  std::vector<Tensor> w_hh;
  std::vector<Tensor> b_ih;
  std::vector<Tensor> b_hh;

  Dropout dropout;

 protected:
  virtual Tensor cell_forward(Tensor input, Tensor state, int64_t layer) = 0;

  RNNOutput CUDNN_forward(Tensor input, Tensor state);
  RNNOutput autograd_forward(Tensor input, Tensor state);

  std::vector<Tensor> flat_weights() const;
  bool use_cudnn(Tensor sample) const;
  Tensor create_dropout_state(Tensor input) const;

  int64_t number_of_gates_;
  bool has_cell_state_;
  at::optional<CuDNNMode> cudnn_mode_;

  // This is copied from pytorch, to determine whether weights are flat for the
  // fast CUDNN route. Otherwise, we have to use non flattened weights, which
  // are much slower.
  // https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/torch/nn/modules/rnn.py#L159-L165
  // TODO Actually since we are in C++ we can probably just actually check if
  // the parameters are flat, instead of relying on data pointers and stuff.
  std::vector<void*> data_ptrs_;
  Tensor flat_weights_;
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

  RNNOptions options;

 private:
  Tensor cell_forward(Tensor input, Tensor state, int64_t layer) override;
  std::function<Tensor(Tensor)> activation_function_;
};

TORCH_MODULE(RNN);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using LSTMOptions = detail::RNNOptionsBase;

class LSTMImpl : public detail::RNNImplBase<LSTMImpl> {
 public:
  explicit LSTMImpl(LSTMOptions options);

 private:
  Tensor cell_forward(Tensor input, Tensor state, int64_t layer) override;
};

TORCH_MODULE(LSTM);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using GRUOptions = detail::RNNOptionsBase;

class GRUImpl : public detail::RNNImplBase<GRUImpl> {
 public:
  explicit GRUImpl(GRUOptions options);

 private:
  Tensor cell_forward(Tensor input, Tensor state, int64_t layer) override;
};

TORCH_MODULE(GRU);

} // namespace nn
} // namespace torch
