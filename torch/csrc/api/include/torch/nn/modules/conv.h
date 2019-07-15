#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional convolution module.
template <size_t D>
struct ConvOptions {
  ConvOptions(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<D> kernel_size) :
                input_channels_(input_channels),
                output_channels_(output_channels),
                kernel_size_(std::move(kernel_size)) {}

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, input_channels);

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, output_channels);

  /// The kernel size to use.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// The stride of the convolution.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// The padding to add to the input volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// The kernel dilation.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// For transpose convolutions, the padding to add to output volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// If true, convolutions will be transpose convolutions (a.k.a.
  /// deconvolutions).
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, transposed) = false;

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, with_bias) = true;

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1;
};

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

/// `ConvOptions` specialized for 1-D convolution.
using Conv1dOptions = ConvOptions<1>;

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

/// `ConvOptions` specialized for 2-D convolution.
using Conv2dOptions = ConvOptions<2>;

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

/// `ConvOptions` specialized for 3-D convolution.
using Conv3dOptions = ConvOptions<3>;

/// A `ModuleHolder` subclass for `Conv3dImpl`.
/// See the documentation for `Conv3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv3d);

} // namespace nn
} // namespace torch
