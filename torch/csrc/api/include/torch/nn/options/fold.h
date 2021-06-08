#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `Fold` module.
///
/// Example:
/// ```
/// Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2, 1}).stride(2));
/// ```
struct TORCH_API FoldOptions {
  FoldOptions(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      // NOLINTNEXTLINE(performance-move-const-arg)
      : output_size_(std::move(output_size)),
        // NOLINTNEXTLINE(performance-move-const-arg)
        kernel_size_(std::move(kernel_size)) {}

  /// describes the spatial shape of the large containing tensor of the sliding
  /// local blocks. It is useful to resolve the ambiguity when multiple input
  /// shapes map to same number of sliding blocks, e.g., with stride > 0.
  TORCH_ARG(ExpandingArray<2>, output_size);

  /// the size of the sliding blocks
  TORCH_ARG(ExpandingArray<2>, kernel_size);

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  TORCH_ARG(ExpandingArray<2>, dilation) = 1;

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  TORCH_ARG(ExpandingArray<2>, padding) = 0;

  /// controls the stride for the sliding blocks.
  TORCH_ARG(ExpandingArray<2>, stride) = 1;
};

namespace functional {
/// Options for `torch::nn::functional::fold`.
///
/// See the documentation for `torch::nn::FoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
/// ```
using FoldFuncOptions = FoldOptions;
} // namespace functional

// ============================================================================

/// Options for the `Unfold` module.
///
/// Example:
/// ```
/// Unfold model(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2));
/// ```
struct TORCH_API UnfoldOptions {
  UnfoldOptions(ExpandingArray<2> kernel_size)
      // NOLINTNEXTLINE(performance-move-const-arg)
      : kernel_size_(std::move(kernel_size)) {}

  /// the size of the sliding blocks
  TORCH_ARG(ExpandingArray<2>, kernel_size);

  /// controls the spacing between the kernel points; also known as the à trous
  /// algorithm.
  TORCH_ARG(ExpandingArray<2>, dilation) = 1;

  /// controls the amount of implicit zero-paddings on both sides for padding
  /// number of points for each dimension before reshaping.
  TORCH_ARG(ExpandingArray<2>, padding) = 0;

  /// controls the stride for the sliding blocks.
  TORCH_ARG(ExpandingArray<2>, stride) = 1;
};

namespace functional {
/// Options for `torch::nn::functional::unfold`.
///
/// See the documentation for `torch::nn::UnfoldOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
using UnfoldFuncOptions = UnfoldOptions;
} // namespace functional

} // namespace nn
} // namespace torch
