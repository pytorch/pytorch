#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) convolution modules.
template <size_t D, typename Derived>
class TORCH_API ConvImpl : public torch::nn::Cloneable<Derived> {
 public:
  ConvImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<D> kernel_size)
      : ConvImpl(ConvOptions<D>(input_channels, output_channels, kernel_size)) {
  }
  explicit ConvImpl(ConvOptions<D> options);

  void reset() override;

  /// Pretty prints the `Conv{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ConvOptions<D> options;

  /// The learned kernel (or "weight").
  Tensor weight;

  /// The learned bias. Only defined if the `with_bias` option was true.
  Tensor bias;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv1d to learn about
/// the exact behavior of this module.
class TORCH_API Conv1dImpl : public ConvImpl<1, Conv1dImpl> {
 public:
  using ConvImpl<1, Conv1dImpl>::ConvImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv1dImpl`.
/// See the documentation for `Conv1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d to learn about
/// the exact behavior of this module.
class TORCH_API Conv2dImpl : public ConvImpl<2, Conv2dImpl> {
 public:
  using ConvImpl<2, Conv2dImpl>::ConvImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv2dImpl`.
/// See the documentation for `Conv2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv3d to learn about
/// the exact behavior of this module.
class TORCH_API Conv3dImpl : public ConvImpl<3, Conv3dImpl> {
 public:
  using ConvImpl<3, Conv3dImpl>::ConvImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv3dImpl`.
/// See the documentation for `Conv3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv3d);

} // namespace nn
} // namespace torch
