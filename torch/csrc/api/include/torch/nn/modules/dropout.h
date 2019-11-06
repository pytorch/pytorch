#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/dropout.h>
#include <torch/nn/functional/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

template <size_t D, typename Derived>
class TORCH_API DropoutImplBase : public torch::nn::Cloneable<Derived> {
 public:
  DropoutImplBase(double p)
      : DropoutImplBase((DropoutOptions<D>)(p)) {}
  explicit DropoutImplBase(const DropoutOptionsBase<D>& options_);

  void reset() override;

  /// Pretty prints the `Dropout{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  DropoutOptionsBase<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies [Dropout](https://arxiv.org/abs/1207.0580) during training.
///
/// See https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout to learn more
/// about the exact semantics of this module.
class TORCH_API DropoutImpl : public DropoutImplBase<1, DropoutImpl> {
 public:
  using DropoutImplBase<1, DropoutImpl>::DropoutImplBase;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `DropoutImpl`.
/// See the documentation for `DropoutImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Dropout2d to learn
/// about the exact behavior of this module.
class TORCH_API Dropout2dImpl : public DropoutImplBase<2, Dropout2dImpl> {
 public:
  using DropoutImplBase<2, Dropout2dImpl>::DropoutImplBase;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Dropout2dImpl`.
/// See the documentation for `Dropout2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Dropout3d to learn
/// about the exact behavior of this module.
class TORCH_API Dropout3dImpl : public DropoutImplBase<3, Dropout3dImpl> {
 public:
  using DropoutImplBase<3, Dropout3dImpl>::DropoutImplBase;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Dropout3dImpl`.
/// See the documentation for `Dropout3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FeatureDropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies spatial [Dropout](https://arxiv.org/abs/1207.0580) to inputs with
/// 2-D or 3-D features.
///
/// The equivalent in Python is
/// [Dropout2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout2d) for
/// 2-D features and
/// [Dropout3d](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout3d) for
/// 3-D features. This `FeatureDropout` module can instead deal with both 2-D
/// and 3-D features.
class TORCH_API FeatureDropoutImpl
    : public DropoutImplBase<2, FeatureDropoutImpl> {
 public:
  FeatureDropoutImpl(double p)
      : FeatureDropoutImpl(FeatureDropoutOptions(p)) {}
  explicit FeatureDropoutImpl(const FeatureDropoutOptions& options_);

  /// During training, applies a noise mask to the input tensor.
  /// During evaluation, applies an identity function.
  Tensor forward(const Tensor& input);

  /// Pretty prints the `FeatureDropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `FeatureDropoutImpl`.
/// See the documentation for `FeatureDropoutImpl` class to learn what methods
/// it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(FeatureDropout);
} // namespace nn
} // namespace torch
