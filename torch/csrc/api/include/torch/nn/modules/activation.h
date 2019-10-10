#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/functional/activation.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies elu over a given input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ELU to learn
/// about the exact behavior of this module.
class TORCH_API ELUImpl : public torch::nn::Cloneable<ELUImpl> {
 public:
  explicit ELUImpl(const ELUOptions& options_ = {});

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `ELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ELUOptions options;
};

TORCH_MODULE(ELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SELU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the selu function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.SELU to learn
/// about the exact behavior of this module.
class TORCH_API SELUImpl : public torch::nn::Cloneable<SELUImpl> {
 public:
  explicit SELUImpl(const SELUOptions& options_ = {});

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `SELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  SELUOptions options;
};

TORCH_MODULE(SELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the hard shrinkage function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Hardshrink to learn
/// about the exact behavior of this module.
class TORCH_API HardshrinkImpl : public torch::nn::Cloneable<HardshrinkImpl> {
 public:
  explicit HardshrinkImpl(const HardshrinkOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Hardshrink` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardshrinkOptions options;
};

TORCH_MODULE(Hardshrink);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardtanh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the HardTanh function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Hardtanh to learn
/// about the exact behavior of this module.
class TORCH_API HardtanhImpl : public torch::nn::Cloneable<HardtanhImpl> {
 public:
  explicit HardtanhImpl(const HardtanhOptions& options_ = {});

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `Hardtanh` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  HardtanhOptions options;
};

TORCH_MODULE(Hardtanh);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LeakyReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LeakyReLU to learn
/// about the exact behavior of this module.
class TORCH_API LeakyReLUImpl : public torch::nn::Cloneable<LeakyReLUImpl> {
 public:
  explicit LeakyReLUImpl(const LeakyReLUOptions& options_ = {});

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `LeakyReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  LeakyReLUOptions options;
};

TORCH_MODULE(LeakyReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the LogSigmoid function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LogSigmoid to learn
/// about the exact behavior of this module.
class TORCH_API LogSigmoidImpl : public torch::nn::Cloneable<LogSigmoidImpl> {
 public:
  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LogSigmoid` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(LogSigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the Softmax function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Softmax to learn
/// about the exact behavior of this module.
class TORCH_API SoftmaxImpl : public torch::nn::Cloneable<SoftmaxImpl> {
 public:
  explicit SoftmaxImpl(int64_t dim) : SoftmaxImpl(SoftmaxOptions(dim)) {}
  explicit SoftmaxImpl(const SoftmaxOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `Softmax` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  SoftmaxOptions options;
};

TORCH_MODULE(Softmax);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the PReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.PReLU to learn
/// about the exact behavior of this module.
class TORCH_API PReLUImpl : public torch::nn::Cloneable<PReLUImpl> {
 public:
  explicit PReLUImpl(const PReLUOptions& options_ = {});

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `PReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  PReLUOptions options;

  /// The learned weight.
  Tensor weight;
};

TORCH_MODULE(PReLU);

} // namespace nn
} // namespace torch
