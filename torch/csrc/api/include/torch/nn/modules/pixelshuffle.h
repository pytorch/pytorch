#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/pixelshuffle.h>
#include <torch/nn/options/pixelshuffle.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelShuffle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
/// to a tensor of shape :math:`(*, C, H \times r, W \times r)`.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.PixelShuffle to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelShuffleOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```
struct TORCH_API PixelShuffleImpl : public torch::nn::Cloneable<PixelShuffleImpl> {
  explicit PixelShuffleImpl(const PixelShuffleOptions& options_);

  /// Pretty prints the `PixelShuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelShuffleOptions options;
};

/// A `ModuleHolder` subclass for `PixelShuffleImpl`.
/// See the documentation for `PixelShuffleImpl` class to learn what methods it
/// provides, and examples of how to use `PixelShuffle` with `torch::nn::PixelShuffleOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(PixelShuffle);

} // namespace nn
} // namespace torch
