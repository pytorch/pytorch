#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

namespace detail {

template <typename Derived>
class _DropoutNd : public torch::nn::Cloneable<Derived> {
 public:
  _DropoutNd(double p) : _DropoutNd(DropoutOptions().p(p)) {};

  explicit _DropoutNd(const DropoutOptions& options_ = {})
    : options(options_) {
    reset();
  }

  void reset() override {
    TORCH_CHECK(
        options.p() >= 0. && options.p() <= 1.,
        "dropout probability has to be between 0 and 1, but got ",
        options.p());
  }

  /// The options with which this `Module` was constructed.
  DropoutOptions options;
};

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Dropout to learn
/// about the exact behavior of this module.
class TORCH_API DropoutImpl : public detail::_DropoutNd<DropoutImpl> {
public:
  using detail::_DropoutNd<DropoutImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `DropoutImpl`.
/// See the documentation for `DropoutImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Dropout2d to learn
/// about the exact behavior of this module.
class TORCH_API Dropout2dImpl : public detail::_DropoutNd<Dropout2dImpl> {
public:
  using detail::_DropoutNd<Dropout2dImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Dropout2dImpl`.
/// See the documentation for `Dropout2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Dropout3d to learn
/// about the exact behavior of this module.
class TORCH_API Dropout3dImpl : public detail::_DropoutNd<Dropout3dImpl> {
public:
  using detail::_DropoutNd<Dropout3dImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout3d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Dropout3dImpl`.
/// See the documentation for `Dropout3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FeatureDropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    : public detail::_DropoutNd<FeatureDropoutImpl> {
 public:
  FeatureDropoutImpl(double p);

  explicit FeatureDropoutImpl(const FeatureDropoutOptions& options_ = {});

  /// During training, applies a noise mask to the input tensor.
  /// During evaluation, applies an identity function.
  Tensor forward(const Tensor& input);

  /// Pretty prints the `FeatureDropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(FeatureDropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AlphaDropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TORCH_API AlphaDropoutImpl
    : public detail::_DropoutNd<AlphaDropoutImpl> {
 public:
  using detail::_DropoutNd<AlphaDropoutImpl>::_DropoutNd;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `AlphaDropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(AlphaDropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FeatureAlphaDropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TORCH_API FeatureAlphaDropoutImpl
    : public detail::_DropoutNd<FeatureAlphaDropoutImpl> {
 public:
  using detail::_DropoutNd<FeatureAlphaDropoutImpl>::_DropoutNd;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `FeatureAlphaDropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(FeatureAlphaDropout);

} // namespace nn
} // namespace torch
