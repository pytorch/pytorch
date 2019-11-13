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
    TORCH_CHECK(
        options.p() >= 0. && options.p() <= 1.,
        "dropout probability has to be between 0 and 1, but got ",
        options.p());
  }

  void reset() override {}

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
class TORCH_API Dropout2dImpl : public detail::_DropoutNd<DropoutImpl> {
public:
  using detail::_DropoutNd<DropoutImpl>::_DropoutNd;

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
class TORCH_API Dropout3dImpl : public detail::_DropoutNd<DropoutImpl> {
public:
  using detail::_DropoutNd<DropoutImpl>::_DropoutNd;

  Tensor forward(Tensor input);

  /// Pretty prints the `Dropout3d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// A `ModuleHolder` subclass for `Dropout3dImpl`.
/// See the documentation for `Dropout3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Dropout3d);

} // namespace nn
} // namespace torch
