#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a fold module.
struct FoldOptions {
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

/// Applies fold over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Fold to learn about
/// the exact behavior of this module.
class TORCH_API FoldImpl : public torch::nn::Cloneable<FoldImpl> {
 public:
  FoldImpl(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : FoldImpl(FoldOptions(output_size, kernel_size)) {}
  explicit FoldImpl(FoldOptions options) : options(std::move(options)) {}

  void reset() override {}

  /// Pretty prints the `Fold` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::Fold";
  }

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FoldOptions options;
};

/// A `ModuleHolder` subclass for `FoldImpl`.
/// See the documentation for `FoldImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Fold);

} // namespace nn
} // namespace torch
