#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

namespace detail {

typedef std::variant<
    enumtype::kZeros,
    enumtype::kReflect,
    enumtype::kReplicate,
    enumtype::kCircular>
    conv_padding_mode_t;

template <size_t D>
using conv_padding_t =
    std::variant<ExpandingArray<D>, enumtype::kValid, enumtype::kSame>;

/// Options for a `D`-dimensional convolution or convolution transpose module.
template <size_t D>
struct ConvNdOptions {
  using padding_t = conv_padding_t<D>;
  ConvNdOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size)
      : in_channels_(in_channels),
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
  TORCH_ARG(padding_t, padding) = 0;

 public:
  decltype(auto) padding(std::initializer_list<int64_t> il) {
    return padding(IntArrayRef{il});
  }

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

  /// Accepted values `torch::kZeros`, `torch::kReflect`, `torch::kReplicate` or
  /// `torch::kCircular`. Default: `torch::kZeros`
  TORCH_ARG(conv_padding_mode_t, padding_mode) = torch::kZeros;
};

} // namespace detail

// ============================================================================

/// Options for a `D`-dimensional convolution module.
template <size_t D>
struct ConvOptions {
  using padding_mode_t = detail::conv_padding_mode_t;
  using padding_t = detail::conv_padding_t<D>;

  ConvOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size)
      : in_channels_(in_channels),
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
  TORCH_ARG(padding_t, padding) = 0;

 public:
  decltype(auto) padding(std::initializer_list<int64_t> il) {
    return padding(IntArrayRef{il});
  }

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

  /// Accepted values `torch::kZeros`, `torch::kReflect`, `torch::kReplicate` or
  /// `torch::kCircular`. Default: `torch::kZeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;
};

/// `ConvOptions` specialized for the `Conv1d` module.
///
/// Example:
/// ```
/// Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false));
/// ```
using Conv1dOptions = ConvOptions<1>;

/// `ConvOptions` specialized for the `Conv2d` module.
///
/// Example:
/// ```
/// Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false));
/// ```
using Conv2dOptions = ConvOptions<2>;

/// `ConvOptions` specialized for the `Conv3d` module.
///
/// Example:
/// ```
/// Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
/// ```
using Conv3dOptions = ConvOptions<3>;

// ============================================================================

namespace functional {

/// Options for a `D`-dimensional convolution functional.
template <size_t D>
struct ConvFuncOptions {
  using padding_t = torch::nn::detail::conv_padding_t<D>;

  /// optional bias of shape `(out_channels)`. Default: ``None``
  TORCH_ARG(torch::Tensor, bias) = Tensor();

  /// The stride of the convolving kernel.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// Implicit paddings on both sides of the input.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(padding_t, padding) = 0;

 public:
  decltype(auto) padding(std::initializer_list<int64_t> il) {
    return padding(IntArrayRef{il});
  }

  /// The spacing between kernel elements.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// Split input into groups, `in_channels` should be divisible by
  /// the number of groups.
  TORCH_ARG(int64_t, groups) = 1;
};

/// `ConvFuncOptions` specialized for `torch::nn::functional::conv1d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
/// ```
using Conv1dFuncOptions = ConvFuncOptions<1>;

/// `ConvFuncOptions` specialized for `torch::nn::functional::conv2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
/// ```
using Conv2dFuncOptions = ConvFuncOptions<2>;

/// `ConvFuncOptions` specialized for `torch::nn::functional::conv3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
/// ```
using Conv3dFuncOptions = ConvFuncOptions<3>;

} // namespace functional

// ============================================================================

template <size_t D>
struct ConvTransposeOptions {
  using padding_mode_t = detail::conv_padding_mode_t;

  ConvTransposeOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size)
      : in_channels_(in_channels),
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

  /// Accepted values `torch::kZeros`, `torch::kReflect`, `torch::kReplicate` or
  /// `torch::kCircular`. Default: `torch::kZeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;
};

/// `ConvTransposeOptions` specialized for the `ConvTranspose1d` module.
///
/// Example:
/// ```
/// ConvTranspose1d model(ConvTranspose1dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
using ConvTranspose1dOptions = ConvTransposeOptions<1>;

/// `ConvTransposeOptions` specialized for the `ConvTranspose2d` module.
///
/// Example:
/// ```
/// ConvTranspose2d model(ConvTranspose2dOptions(3, 2,
/// 3).stride(1).bias(false));
/// ```
using ConvTranspose2dOptions = ConvTransposeOptions<2>;

/// `ConvTransposeOptions` specialized for the `ConvTranspose3d` module.
///
/// Example:
/// ```
/// ConvTranspose3d model(ConvTranspose3dOptions(2, 2,
/// 2).stride(1).bias(false));
/// ```
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

  /// Additional size added to one side of each dimension in the output shape.
  /// Default: 0
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// Split input into groups, `in_channels` should be divisible by
  /// the number of groups.
  TORCH_ARG(int64_t, groups) = 1;

  /// The spacing between kernel elements.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;
};

/// `ConvTransposeFuncOptions` specialized for
/// `torch::nn::functional::conv_transpose1d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
/// ```
using ConvTranspose1dFuncOptions = ConvTransposeFuncOptions<1>;

/// `ConvTransposeFuncOptions` specialized for
/// `torch::nn::functional::conv_transpose2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
/// ```
using ConvTranspose2dFuncOptions = ConvTransposeFuncOptions<2>;

/// `ConvTransposeFuncOptions` specialized for
/// `torch::nn::functional::conv_transpose3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose3d(x, weight, F::ConvTranspose3dFuncOptions().stride(1));
/// ```
using ConvTranspose3dFuncOptions = ConvTransposeFuncOptions<3>;

} // namespace functional

} // namespace nn
} // namespace torch
