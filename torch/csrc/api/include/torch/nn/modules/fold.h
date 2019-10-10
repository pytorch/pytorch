#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/options/fold.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Applies fold over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Fold to learn about
/// the exact behavior of this module.
class TORCH_API FoldImpl : public torch::nn::Cloneable<FoldImpl> {
 public:
  FoldImpl(ExpandingArray<2> output_size, ExpandingArray<2> kernel_size)
      : FoldImpl(FoldOptions(output_size, kernel_size)) {}
  explicit FoldImpl(const FoldOptions& options_);

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
