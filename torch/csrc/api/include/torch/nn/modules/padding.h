#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/padding.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) ReflectionPad modules.
template <size_t D, typename Derived>
class TORCH_API ReflectionPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ReflectionPadImpl(ExpandingArray<D*2> padding)
      : ReflectionPadImpl(ReflectionPadOptions<D>(padding)) {}
  explicit ReflectionPadImpl(const ReflectionPadOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `ReflectionPad{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReflectionPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad1d to learn
/// about the exact behavior of this module.
class TORCH_API ReflectionPad1dImpl : public ReflectionPadImpl<1, ReflectionPad1dImpl> {
 public:
  using ReflectionPadImpl<1, ReflectionPad1dImpl>::ReflectionPadImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `ReflectionPad1dImpl`.
/// See the documentation for `ReflectionPad1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ReflectionPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad2d to learn
/// about the exact behavior of this module.
class TORCH_API ReflectionPad2dImpl : public ReflectionPadImpl<2, ReflectionPad2dImpl> {
 public:
  using ReflectionPadImpl<2, ReflectionPad2dImpl>::ReflectionPadImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `ReflectionPad2dImpl`.
/// See the documentation for `ReflectionPad2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ReflectionPad2d);

} // namespace nn
} // namespace torch
