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
  ELUImpl() : ELUImpl(ELUOptions()) {}
  explicit ELUImpl(const ELUOptions& options_);

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `ELU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ELUOptions options;
};

TORCH_MODULE(ELU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hardshrink ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the hard shrinkage function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Hardshrink to learn
/// about the exact behavior of this module.
class TORCH_API HardshrinkImpl : public torch::nn::Cloneable<HardshrinkImpl> {
 public:
  HardshrinkImpl() : HardshrinkImpl(HardshrinkOptions()) {}
  explicit HardshrinkImpl(const HardshrinkOptions& options_);

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
  HardtanhImpl() : HardtanhImpl(HardtanhOptions()) {}
  explicit HardtanhImpl(const HardtanhOptions& options_);

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
  LeakyReLUImpl() : LeakyReLUImpl(LeakyReLUOptions()) {}
  explicit LeakyReLUImpl(const LeakyReLUOptions& options_);

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
  LogSigmoidImpl() {}

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LogSigmoid` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(LogSigmoid);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the PReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.PReLU to learn
/// about the exact behavior of this module.
class TORCH_API PReLUImpl : public torch::nn::Cloneable<PReLUImpl> {
 public:
  PReLUImpl() : PReLUImpl(PReLUOptions()) {}
  explicit PReLUImpl(const PReLUOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `PReLUImpl` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  PReLUOptions options;

  /// The learned weight.
  Tensor weight;
};

TORCH_MODULE(PReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ReLU function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReLU to learn
/// about the exact behavior of this module.
class TORCH_API ReLUImpl : public torch::nn::Cloneable<ReLUImpl> {
 public:
  ReLUImpl() : ReLUImpl(ReLUOptions()) {}
  explicit ReLUImpl(const ReLUOptions& options_);

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `ReLU` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLUOptions options;
};

TORCH_MODULE(ReLU);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ReLU6 function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReLU6 to learn
/// about the exact behavior of this module.
class TORCH_API ReLU6Impl : public torch::nn::Cloneable<ReLU6Impl> {
 public:
  ReLU6Impl() : ReLU6Impl(ReLU6Options()) {}
  explicit ReLU6Impl(const ReLU6Options& options_);

  Tensor forward(Tensor& input);

  void reset() override;

  /// Pretty prints the `ReLU6` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReLU6Options options;
};

TORCH_MODULE(ReLU6);

} // namespace nn
} // namespace torch
