#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>
#include <ATen/core/Error.h>
#include <ATen/core/optional.h>

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
  TORCH_ARG(bool, bidirectional) = false;
  TORCH_ARG(bool, batch_first) = false;
};

template <typename Derived>
class RNNImplBase : public torch::nn::Cloneable<Derived> {
 public:
  // These must line up with the CUDNN mode codes:
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
  enum class CuDNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

  explicit RNNImplBase(
      RNNOptionsBase options_,
      at::optional<CuDNNMode> cudnn_mode = at::nullopt,
      int64_t number_of_gates = 1);

  /// Initializes the parameters of the RNN module.
  void reset() override;

  /// Overrides `nn::Module::to()` to call `flatten_parameters()` after the
  /// original operation.
  void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false)
      override;
  void to(torch::Dtype dtype, bool non_blocking = false) override;
  void to(torch::Device device, bool non_blocking = false) override;

  /// Modifies the internal storage of weights for optimization purposes.
  ///
  /// On CPU, this method should be called if any of the weight or bias vectors
  /// are changed. On GPU, it should be called __any time the storage of any
  /// parameter is modified__, e.g. any time a parameter is assigned a new
  /// value. This allows using the fast path in cuDNN implementations of
  /// respective RNN `forward()` methods. It is called once upon construction,
  /// inside `reset()`.
  void flatten_parameters();

  /// Returns the cuDNN mode of the RNN subclass if it has one. See
  /// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
  /// for a list of all cuDNN modes.
  at::optional<CuDNNMode> cudnn_mode() const noexcept;

  RNNOptionsBase options;

  /// The weights for `input x hidden` gates.
  std::vector<Tensor> w_ih;
  /// The weights for `hidden x hidden` gates.
  std::vector<Tensor> w_hh;
  /// The biases for `input x hidden` gates.
  std::vector<Tensor> b_ih;
  /// The biases for `hidden x hidden` gates.
  std::vector<Tensor> b_hh;

  /// The dropout module, if dropout is used.
  Dropout dropout{nullptr};

 protected:
  /// The function signature of `at::lstm`, `at::rnn_relu`, `at::gru` etc.
  using RNNFunctionSignature = std::tuple<Tensor, Tensor>(
      /*input=*/const Tensor&,
      /*state=*/const Tensor&,
      /*params=*/TensorList,
      /*has_biases=*/bool,
      /*layers=*/int64_t,
      /*dropout=*/double,
      /*train=*/bool,
      /*bidirectional=*/bool,
      /*batch_first=*/bool);

  RNNOutput generic_forward(
      std::function<RNNFunctionSignature> function,
      Tensor input,
      Tensor state);

  /// Returns a flat vector of all weights, with layer weights following each
  /// other sequentially in (w_ih, w_hh, b_ih, b_hh) order.
  std::vector<Tensor> flat_weights() const;

  /// Returns true if any of the parameters (weights, biases) alias each other.
  bool any_parameters_alias() const;

  /// The number of gate weights/biases required by the RNN subclass.
  int64_t number_of_gates_;

  /// The cuDNN RNN mode, if this RNN subclass has any.
  at::optional<CuDNNMode> cudnn_mode_;

  /// The cached result of the latest `flat_weights()` call.
  std::vector<Tensor> flat_weights_;
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
  TORCH_ARG(bool, bidirectional) = false;
  TORCH_ARG(bool, batch_first) = false;
  TORCH_ARG(RNNActivation, activation) = RNNActivation::ReLU;
};

class RNNImpl : public detail::RNNImplBase<RNNImpl> {
 public:
  RNNImpl(int64_t input_size, int64_t hidden_size)
      : RNNImpl(RNNOptions(input_size, hidden_size)) {}
  explicit RNNImpl(RNNOptions options);

  RNNOutput forward(Tensor input, Tensor state = {});

  RNNOptions options;
};

/// A multi-layer Elman RNN module with Tanh or ReLU activation.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.RNN for more
/// documenation.
TORCH_MODULE(RNN);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using LSTMOptions = detail::RNNOptionsBase;

class LSTMImpl : public detail::RNNImplBase<LSTMImpl> {
 public:
  LSTMImpl(int64_t input_size, int64_t hidden_size)
      : LSTMImpl(LSTMOptions(input_size, hidden_size)) {}
  explicit LSTMImpl(LSTMOptions options);

  RNNOutput forward(Tensor input, Tensor state = {});
};

/// A multi-layer long-short-term-memory (LSTM) module.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LSTM for more
/// documenation.
TORCH_MODULE(LSTM);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using GRUOptions = detail::RNNOptionsBase;

class GRUImpl : public detail::RNNImplBase<GRUImpl> {
 public:
  GRUImpl(int64_t input_size, int64_t hidden_size)
      : GRUImpl(GRUOptions(input_size, hidden_size)) {}
  explicit GRUImpl(GRUOptions options);

  RNNOutput forward(Tensor input, Tensor state = {});
};

/// A multi-layer gated recurrent unit (GRU) module.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.GRU for more
/// documenation.
TORCH_MODULE(GRU);

} // namespace nn
} // namespace torch
