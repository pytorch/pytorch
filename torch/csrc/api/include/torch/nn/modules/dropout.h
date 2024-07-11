#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/Export.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

namespace detail {

template <typename Derived>
class _DropoutNd : public torch::nn::Cloneable<Derived> {
 public:
  _DropoutNd(double p) : _DropoutNd(DropoutOptions().p(p)){};

  explicit _DropoutNd(const DropoutOptions& options_ = {}) : options(options_) {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
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

} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 1-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Dropout to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::DropoutOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Dropout model(DropoutOptions().p(0.42).inplace(true));
/// ```
class TORCH_API DropoutImpl : public detail::_DropoutNd<DropoutImpl> {
 public:
  using detail::_DropoutNd<DropoutImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `DropoutImpl`.
/// See the documentation for `DropoutImpl` class to learn what methods it
/// provides, and examples of how to use `Dropout` with
/// `torch::nn::DropoutOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Dropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 2-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Dropout2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::Dropout2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Dropout2d model(Dropout2dOptions().p(0.42).inplace(true));
/// ```
class TORCH_API Dropout2dImpl : public detail::_DropoutNd<Dropout2dImpl> {
 public:
  using detail::_DropoutNd<Dropout2dImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Dropout2dImpl`.
/// See the documentation for `Dropout2dImpl` class to learn what methods it
/// provides, and examples of how to use `Dropout2d` with
/// `torch::nn::Dropout2dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Dropout2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies dropout over a 3-D input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Dropout3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::Dropout3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Dropout3d model(Dropout3dOptions().p(0.42).inplace(true));
/// ```
class TORCH_API Dropout3dImpl : public detail::_DropoutNd<Dropout3dImpl> {
 public:
  using detail::_DropoutNd<Dropout3dImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout3d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Dropout3dImpl`.
/// See the documentation for `Dropout3dImpl` class to learn what methods it
/// provides, and examples of how to use `Dropout3d` with
/// `torch::nn::Dropout3dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Dropout3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AlphaDropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Alpha Dropout over the input.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.AlphaDropout to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AlphaDropoutOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AlphaDropout model(AlphaDropoutOptions(0.2).inplace(true));
/// ```
class TORCH_API AlphaDropoutImpl : public detail::_DropoutNd<AlphaDropoutImpl> {
 public:
  using detail::_DropoutNd<AlphaDropoutImpl>::_DropoutNd;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `AlphaDropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `AlphaDropoutImpl`.
/// See the documentation for `AlphaDropoutImpl` class to learn what methods it
/// provides, and examples of how to use `AlphaDropout` with
/// `torch::nn::AlphaDropoutOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(AlphaDropout);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FeatureAlphaDropout
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// See the documentation for `torch::nn::FeatureAlphaDropoutOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// FeatureAlphaDropout model(FeatureAlphaDropoutOptions(0.2).inplace(true));
/// ```
class TORCH_API FeatureAlphaDropoutImpl
    : public detail::_DropoutNd<FeatureAlphaDropoutImpl> {
 public:
  using detail::_DropoutNd<FeatureAlphaDropoutImpl>::_DropoutNd;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `FeatureAlphaDropout` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `FeatureAlphaDropoutImpl`.
/// See the documentation for `FeatureAlphaDropoutImpl` class to learn what
/// methods it provides, and examples of how to use `FeatureAlphaDropout` with
/// `torch::nn::FeatureAlphaDropoutOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(FeatureAlphaDropout);

} // namespace nn
} // namespace torch
