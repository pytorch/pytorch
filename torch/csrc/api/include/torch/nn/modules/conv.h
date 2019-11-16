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
  explicit ConvImpl(ConvOptions<D> options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `Conv{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ConvOptions<D> options;

  /// The learned kernel (or "weight").
  Tensor weight;

  /// The learned bias. Only defined if the `bias` option was true.
  Tensor bias;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv1d to learn about
/// the exact behavior of this module.
class TORCH_API Conv1dImpl : public ConvImpl<1, Conv1dImpl> {
 public:
  Conv1dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<1> kernel_size)
      : Conv1dImpl(ConvOptions<1>(input_channels, output_channels, kernel_size)) {
  }
  explicit Conv1dImpl(ConvOptions<1> options_);
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
  Conv2dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<2> kernel_size)
      : Conv2dImpl(ConvOptions<2>(input_channels, output_channels, kernel_size)) {
  }
  explicit Conv2dImpl(ConvOptions<2> options_);
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
  Conv3dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<3> kernel_size)
      : Conv3dImpl(ConvOptions<3>(input_channels, output_channels, kernel_size)) {
  }
  explicit Conv3dImpl(ConvOptions<3> options_);
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv3dImpl`.
/// See the documentation for `Conv3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Base class for all (dimension-specialized) convolution transpose modules.
template <size_t D, typename Derived>
class TORCH_API ConvTransposeImplBase : public torch::nn::Cloneable<Derived> {
 protected:
  ExpandingArray<D> _output_padding(
      const Tensor& input, const std::vector<int64_t>& output_size,
      const ExpandingArray<D>& stride, const ExpandingArray<D>& padding,
      const ExpandingArray<D>& kernel_size);

  void reset_parameters();

 public:
  ConvTransposeImplBase(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<D> kernel_size)
      : ConvTransposeImplBase(ConvTransposeOptionsBase<D>(input_channels, output_channels, kernel_size)) {
  }
  explicit ConvTransposeImplBase(const ConvTransposeOptionsBase<D>& options_);

  void reset() override;

  /// Pretty prints the `ConvTranspose{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ConvTransposeOptionsBase<D> options;

  /// The learned kernel (or "weight").
  Tensor weight;

  /// The learned bias. Only defined if the `with_bias` option was true.
  Tensor bias;
};

/// Applies the ConvTranspose1d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose1d to
/// learn about the exact behavior of this module.
class TORCH_API ConvTranspose1dImpl : public ConvTransposeImplBase<1, ConvTranspose1dImpl> {
 public:
  using ConvTransposeImplBase<1, ConvTranspose1dImpl>::ConvTransposeImplBase;

  Tensor forward(const Tensor& input,
                 const std::vector<int64_t>& output_size = {});
};

TORCH_MODULE(ConvTranspose1d);

} // namespace nn
} // namespace torch
