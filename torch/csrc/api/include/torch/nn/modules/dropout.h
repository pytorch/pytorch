#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
struct DropoutOptions {
  DropoutOptions(double rate);
  /// The probability with which a particular component of the input is set to
  /// zero.
  TORCH_ARG(double, rate) = 0.5;
};

namespace detail {
template <typename Derived>
class DropoutImplBase : public torch::nn::Cloneable<Derived> {
 public:
  explicit DropoutImplBase(double rate)
      : DropoutImplBase(DropoutOptions(rate)) {}
  explicit DropoutImplBase(DropoutOptions options_);

  void reset() override;

  /// During training, applies a noise mask to the input tensor.
  /// During evaluation, applies an identity function.
  Tensor forward(Tensor input);

  /// Returns a noise mask that can be applied to the given input tensor.
  /// Used inside `forward()` to generate the noise mask for dropout.
  virtual Tensor noise_mask(Tensor input) const = 0;

  DropoutOptions options;
};
} // namespace detail

/// Applies [Dropout](https://arxiv.org/abs/1207.0580) during training.
///
/// See https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout to learn more
/// about the exact semantics of this module.
class DropoutImpl : public detail::DropoutImplBase<DropoutImpl> {
 public:
  using detail::DropoutImplBase<DropoutImpl>::DropoutImplBase;
  Tensor noise_mask(Tensor input) const override;
};

/// Applies [Dropout](https://arxiv.org/abs/1207.0580) to inputs with
/// 2-dimensional features.
///
/// See https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout2d to learn more
/// about the exact semantics of this module.
class Dropout2dImpl : public detail::DropoutImplBase<Dropout2dImpl> {
 public:
  using detail::DropoutImplBase<Dropout2dImpl>::DropoutImplBase;
  Tensor noise_mask(Tensor input) const override;
};

/// A `ModuleHolder` subclass for `DropoutImpl`.
/// See the documentation for `DropoutImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout);

/// A `ModuleHolder` subclass for `Dropout2dImpl`.
/// See the documentation for `Dropout2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout2d);
} // namespace nn
} // namespace torch
