#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/rnn.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace nn {

/// The output of a single invocation of an RNN module's `forward()` method.
struct TORCH_API RNNOutput {
  /// The result of applying the specific RNN algorithm
  /// to the input tensor and input state.
  Tensor output;
  /// The new, updated state that can be fed into the RNN
  /// in the next forward step.
  Tensor state;
};

namespace detail {
/// Base class for all RNN implementations (intended for code sharing).
template <typename Derived>
class TORCH_API RNNImplBase : public torch::nn::Cloneable<Derived> {
 public:
  /// These must line up with the CUDNN mode codes:
  /// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
  enum class CuDNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

  explicit RNNImplBase(
      const RNNOptionsBase& options_,
      optional<CuDNNMode> cudnn_mode = nullopt,
      int64_t number_of_gates = 1);

  /// Initializes the parameters of the RNN module.
  void reset() override;

  /// Overrides `nn::Module::to()` to call `flatten_parameters()` after the
  /// original operation.
  void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false)
      override;
  void to(torch::Dtype dtype, bool non_blocking = false) override;
  void to(torch::Device device, bool non_blocking = false) override;

  /// Pretty prints the RNN module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Modifies the internal storage of weights for optimization purposes.
  ///
  /// On CPU, this method should be called if any of the weight or bias vectors
  /// are changed (i.e. weights are added or removed). On GPU, it should be
  /// called __any time the storage of any parameter is modified__, e.g. any
  /// time a parameter is assigned a new value. This allows using the fast path
  /// in cuDNN implementations of respective RNN `forward()` methods. It is
  /// called once upon construction, inside `reset()`.
  void flatten_parameters();

  /// The RNN's options.
  RNNOptionsBase options;

  /// The weights for `input x hidden` gates.
  std::vector<Tensor> w_ih;
  /// The weights for `hidden x hidden` gates.
  std::vector<Tensor> w_hh;
  /// The biases for `input x hidden` gates.
  std::vector<Tensor> b_ih;
  /// The biases for `hidden x hidden` gates.
  std::vector<Tensor> b_hh;

 protected:
  /// The function signature of `rnn_relu`, `rnn_tanh` and `gru`.
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

  /// A generic `forward()` used for RNN and GRU (but not LSTM!). Takes the ATen
  /// RNN function as first argument.
  RNNOutput generic_forward(
      std::function<RNNFunctionSignature> function,
      const Tensor& input,
      Tensor state);

  /// Returns a flat vector of all weights, with layer weights following each
  /// other sequentially in (w_ih, w_hh, b_ih, b_hh) order.
  std::vector<Tensor> flat_weights() const;

  /// Very simple check if any of the parameters (weights, biases) are the same.
  bool any_parameters_alias() const;

  /// The number of gate weights/biases required by the RNN subclass.
  int64_t number_of_gates_;

  /// The cuDNN RNN mode, if this RNN subclass has any.
  optional<CuDNNMode> cudnn_mode_;

  /// The cached result of the latest `flat_weights()` call.
  std::vector<Tensor> flat_weights_;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer Elman RNN module with Tanh or ReLU activation.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.RNN to learn about the
/// exact behavior of this module.
class TORCH_API RNNImpl : public detail::RNNImplBase<RNNImpl> {
 public:
  RNNImpl(int64_t input_size, int64_t hidden_size)
      : RNNImpl(RNNOptions(input_size, hidden_size)) {}
  explicit RNNImpl(const RNNOptions& options_);

  /// Pretty prints the `RNN` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies the `RNN` module to an input sequence and input state.
  /// The `input` should follow a `(sequence, batch, features)` layout unless
  /// `batch_first` is true, in which case the layout should be `(batch,
  /// sequence, features)`.
  RNNOutput forward(const Tensor& input, Tensor state = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})
 public:
  RNNOptions options;
};

/// A `ModuleHolder` subclass for `RNNImpl`.
/// See the documentation for `RNNImpl` class to learn what methods it provides,
/// or the documentation for `ModuleHolder` to learn about PyTorch's module
/// storage semantics.
TORCH_MODULE(RNN);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer long-short-term-memory (LSTM) module.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LSTM to learn about the
/// exact behavior of this module.
class TORCH_API LSTMImpl : public detail::RNNImplBase<LSTMImpl> {
 public:
  LSTMImpl(int64_t input_size, int64_t hidden_size)
      : LSTMImpl(LSTMOptions(input_size, hidden_size)) {}
  explicit LSTMImpl(const LSTMOptions& options_);

  /// Applies the `LSTM` module to an input sequence and input state.
  /// The `input` should follow a `(sequence, batch, features)` layout unless
  /// `batch_first` is true, in which case the layout should be `(batch,
  /// sequence, features)`.
  RNNOutput forward(const Tensor& input, Tensor state = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})
};

/// A `ModuleHolder` subclass for `LSTMImpl`.
/// See the documentation for `LSTMImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(LSTM);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A multi-layer gated recurrent unit (GRU) module.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.GRU to learn about the
/// exact behavior of this module.
class TORCH_API GRUImpl : public detail::RNNImplBase<GRUImpl> {
 public:
  GRUImpl(int64_t input_size, int64_t hidden_size)
      : GRUImpl(GRUOptions(input_size, hidden_size)) {}
  explicit GRUImpl(const GRUOptions& options_);

  /// Applies the `GRU` module to an input sequence and input state.
  /// The `input` should follow a `(sequence, batch, features)` layout unless
  /// `batch_first` is true, in which case the layout should be `(batch,
  /// sequence, features)`.
  RNNOutput forward(const Tensor& input, Tensor state = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})
};

/// A `ModuleHolder` subclass for `GRUImpl`.
/// See the documentation for `GRUImpl` class to learn what methods it provides,
/// or the documentation for `ModuleHolder` to learn about PyTorch's module
/// storage semantics.
TORCH_MODULE(GRU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCellImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
/// Base class for all RNNCell implementations (intended for code sharing).
template <typename Derived>
class TORCH_API RNNCellImplBase : public torch::nn::Cloneable<Derived> {
 public:
  explicit RNNCellImplBase(const RNNCellOptionsBase& options_);

  /// Initializes the parameters of the RNNCell module.
  void reset() override;

  void reset_parameters();

  /// Pretty prints the RNN module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  RNNCellOptionsBase options_base;

  Tensor weight_ih;
  Tensor weight_hh;
  Tensor bias_ih;
  Tensor bias_hh;

 protected:
  void check_forward_input(const Tensor& input) const;
  void check_forward_hidden(const Tensor& input, const Tensor& hx, std::string hidden_label) const;
  virtual std::string get_nonlinearity_str() const;
};
} // namespace detail


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// An Elman RNN cell with tanh or ReLU non-linearity.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.RNNCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::RNNCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// RNNCell model(RNNCellOptions(20, 10).bias(false).nonlinearity(torch::kReLU));
/// ```
class TORCH_API RNNCellImpl : public detail::RNNCellImplBase<RNNCellImpl> {
 public:
  RNNCellImpl(int64_t input_size, int64_t hidden_size)
      : RNNCellImpl(RNNCellOptions(input_size, hidden_size)) {}
  explicit RNNCellImpl(const RNNCellOptions& options_);

  Tensor forward(const Tensor& input, Tensor hx = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  RNNCellOptions options;

 protected:
  std::string get_nonlinearity_str() const override;
};

/// A `ModuleHolder` subclass for `RNNCellImpl`.
/// See the documentation for `RNNCellImpl` class to learn what methods it
/// provides, and examples of how to use `RNNCell` with `torch::nn::RNNCellOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(RNNCell);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTMCell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A long short-term memory (LSTM) cell.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LSTMCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LSTMCell model(LSTMCellOptions(20, 10).bias(false));
/// ```
class TORCH_API LSTMCellImpl : public detail::RNNCellImplBase<LSTMCellImpl> {
 public:
  LSTMCellImpl(int64_t input_size, int64_t hidden_size)
      : LSTMCellImpl(LSTMCellOptions(input_size, hidden_size)) {}
  explicit LSTMCellImpl(const LSTMCellOptions& options_);

  std::tuple<Tensor, Tensor> forward(const Tensor& input, torch::optional<std::tuple<Tensor, Tensor>> hx_opt = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(torch::optional<std::tuple<Tensor, Tensor>>())})

 public:
  LSTMCellOptions options;
};

/// A `ModuleHolder` subclass for `LSTMCellImpl`.
/// See the documentation for `LSTMCellImpl` class to learn what methods it
/// provides, and examples of how to use `LSTMCell` with `torch::nn::LSTMCellOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(LSTMCell);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRUCell ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A gated recurrent unit (GRU) cell.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.GRUCell to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::GRUCellOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GRUCell model(GRUCellOptions(20, 10).bias(false));
/// ```
class TORCH_API GRUCellImpl : public detail::RNNCellImplBase<GRUCellImpl> {
 public:
  GRUCellImpl(int64_t input_size, int64_t hidden_size)
      : GRUCellImpl(GRUCellOptions(input_size, hidden_size)) {}
  explicit GRUCellImpl(const GRUCellOptions& options_);

  Tensor forward(const Tensor& input, Tensor hx = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())})

 public:
  GRUCellOptions options;
};

/// A `ModuleHolder` subclass for `GRUCellImpl`.
/// See the documentation for `GRUCellImpl` class to learn what methods it
/// provides, and examples of how to use `GRUCell` with `torch::nn::GRUCellOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(GRUCell);

} // namespace nn
} // namespace torch
