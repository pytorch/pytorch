#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// Options for `Dropout` and `FeatureDropout`.
struct TORCH_API DropoutOptions {
  /* implicit */ DropoutOptions(double rate = 0.5);
  /// The probability with which a particular component of the input is set to
  /// zero.
  /// Changes to this parameter at runtime are effective.
  TORCH_ARG(double, rate);
};

namespace detail {
template <typename Derived>
class DropoutImplBase : public torch::nn::Cloneable<Derived> {
 public:
  explicit DropoutImplBase(DropoutOptions options_ = DropoutOptions());

  void reset() override;

  /// The options used to configure this `Dropout` module.
  DropoutOptions options;
};
} // namespace detail

/// Applies [Dropout](https://arxiv.org/abs/1207.0580) during training.
///
/// See https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout to learn more
/// about the exact semantics of this module.
class TORCH_API DropoutImpl : public detail::DropoutImplBase<DropoutImpl> {
 public:
  using detail::DropoutImplBase<DropoutImpl>::DropoutImplBase;
  /// During training, applies a noise mask to the input tensor.
  /// During evaluation, applies an identity function.
  Tensor forward(Tensor input);
};

/// Applies spatial [Dropout](https://arxiv.org/abs/1207.0580) to inputs with
/// 2-D or 3-D features.
///
/// The equivalent in Python is
/// [Dropout2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout2d) for
/// 2-D features and
/// [Dropout3d](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout3d) for
/// 3-D features. This `FeatureDropout` module can instead deal with both 2-D
/// and 3-D features.
class TORCH_API FeatureDropoutImpl : public detail::DropoutImplBase<FeatureDropoutImpl> {
 public:
  using detail::DropoutImplBase<FeatureDropoutImpl>::DropoutImplBase;
  /// During training, applies a noise mask to the input tensor.
  /// During evaluation, applies an identity function.
  Tensor forward(Tensor input);
};

/// A `ModuleHolder` subclass for `DropoutImpl`.
/// See the documentation for `DropoutImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout);

/// A `ModuleHolder` subclass for `FeatureDropoutImpl`.
/// See the documentation for `FeatureDropoutImpl` class to learn what methods
/// it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(FeatureDropout);
} // namespace nn
} // namespace torch
