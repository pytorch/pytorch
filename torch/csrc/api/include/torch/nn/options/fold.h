#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/nn/options/common.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a fold module.
struct TORCH_API FoldOptions {
  FoldOptions(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : output_size_(std::move(output_size)),
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

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Fold)

// ============================================================================

/// Options for an Unfold functional and module.
struct TORCH_API UnfoldOptions {
  UnfoldOptions(ExpandingArray<2> kernel_size)
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

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(Unfold)

} // namespace nn
} // namespace torch
