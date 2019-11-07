#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

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

/// `ConvOptions` specialized for 1-D convolution.
using Conv1dOptions = ConvOptions<1>;

/// `ConvOptions` specialized for 2-D convolution.
using Conv2dOptions = ConvOptions<2>;

/// `ConvOptions` specialized for 3-D convolution.
using Conv3dOptions = ConvOptions<3>;

} // namespace nn
} // namespace torch
