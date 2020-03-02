#pragma once

#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

namespace detail {

typedef c10::variant<enumtype::kZeros, enumtype::kCircular> conv_padding_mode_t;

/// Options for a `D`-dimensional convolution or convolution transpose module.
template <size_t D>
struct ConvNdOptions {
  ConvNdOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size) :
                in_channels_(in_channels),
                out_channels_(out_channels),
                kernel_size_(std::move(kernel_size)) {}

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels);

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels);

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

  /// If true, convolutions will be transpose convolutions (a.k.a.
  /// deconvolutions).
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, transposed) = false;

  /// For transpose convolutions, the padding to add to output volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1;

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, bias) = true;

  /// Accepted values `zeros` and `circular` Default: `zeros`
  TORCH_ARG(conv_padding_mode_t, padding_mode) = torch::kZeros;
};

} // namespace detail

// ============================================================================

/// Options for a `D`-dimensional convolution module.
template <size_t D>
struct ConvOptions {
  using padding_mode_t = detail::conv_padding_mode_t;

  ConvOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size) :
                in_channels_(in_channels),
                out_channels_(out_channels),
                kernel_size_(std::move(kernel_size)) {}

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels);

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels);

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

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1;

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, bias) = true;

  /// Accepted values `zeros` and `circular` Default: `zeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;
};

/// `ConvOptions` specialized for 1-D convolution.
using Conv1dOptions = ConvOptions<1>;

/// `ConvOptions` specialized for 2-D convolution.
using Conv2dOptions = ConvOptions<2>;

/// `ConvOptions` specialized for 3-D convolution.
using Conv3dOptions = ConvOptions<3>;

// ============================================================================

namespace functional {

/// Options for a `D`-dimensional convolution functional.
template <size_t D>
struct ConvFuncOptions {
  /// optional bias of shape `(out_channels)`. Default: ``None``
  TORCH_ARG(torch::Tensor, bias) = Tensor();

  /// The stride of the convolving kernel.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// Implicit paddings on both sides of the input.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// The spacing between kernel elements.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// Split input into groups, `in_channels` should be divisible by
  /// the number of groups.
  TORCH_ARG(int64_t, groups) = 1;
};

/// `ConvFuncOptions` specialized for 1-D convolution.
using Conv1dFuncOptions = ConvFuncOptions<1>;

/// `ConvFuncOptions` specialized for 2-D convolution.
using Conv2dFuncOptions = ConvFuncOptions<2>;

/// `ConvFuncOptions` specialized for 3-D convolution.
using Conv3dFuncOptions = ConvFuncOptions<3>;

} // namespace functional

// ============================================================================

template <size_t D>
struct ConvTransposeOptions {
  using padding_mode_t = detail::conv_padding_mode_t;

  ConvTransposeOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size) :
                in_channels_(in_channels),
                out_channels_(out_channels),
                kernel_size_(std::move(kernel_size)) {}

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels);

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels);

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

  /// For transpose convolutions, the padding to add to output volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1;

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, bias) = true;

  /// The kernel dilation.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// Accepted values `zeros` and `circular` Default: `zeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;
};

/// `ConvTransposeOptions` specialized for 1-D convolution.
using ConvTranspose1dOptions = ConvTransposeOptions<1>;

/// `ConvTransposeOptions` specialized for 2-D convolution.
using ConvTranspose2dOptions = ConvTransposeOptions<2>;

/// `ConvTransposeOptions` specialized for 3-D convolution.
using ConvTranspose3dOptions = ConvTransposeOptions<3>;

// ============================================================================

namespace functional {

/// Options for a `D`-dimensional convolution functional.
template <size_t D>
struct ConvTransposeFuncOptions {
  /// optional bias of shape `(out_channels)`. Default: ``None``
  TORCH_ARG(torch::Tensor, bias) = Tensor();

  /// The stride of the convolving kernel.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// Implicit paddings on both sides of the input.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// Additional size added to one side of each dimension in the output shape. Default: 0
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// Split input into groups, `in_channels` should be divisible by
  /// the number of groups.
  TORCH_ARG(int64_t, groups) = 1;

  /// The spacing between kernel elements.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;
};

/// `ConvTransposeFuncOptions` specialized for 1-D convolution transpose.
using ConvTranspose1dFuncOptions = ConvTransposeFuncOptions<1>;

/// `ConvTransposeFuncOptions` specialized for 2-D convolution transpose.
using ConvTranspose2dFuncOptions = ConvTransposeFuncOptions<2>;

/// `ConvTransposeFuncOptions` specialized for 3-D convolution transpose.
using ConvTranspose3dFuncOptions = ConvTransposeFuncOptions<3>;

} // namespace functional

} // namespace nn
} // namespace torch
