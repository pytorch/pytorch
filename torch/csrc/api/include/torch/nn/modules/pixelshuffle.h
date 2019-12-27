#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/pixelshuffle.h>
#include <torch/nn/options/pixelshuffle.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
/// to a tensor of shape :math:`(*, C, H \times r, W \times r)`.
/// This is useful for implementing efficient sub-pixel convolution
/// with a stride of :math:`1/r`.
/// Look at the paper:
/// `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
/// by Shi et. al (2016) for more details.
struct TORCH_API PixelShuffleImpl : public torch::nn::Cloneable<PixelShuffleImpl> {
  explicit PixelShuffleImpl(const PixelShuffleOptions& options_);

  /// Pretty prints the `PixelShuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelShuffleOptions options;
};

TORCH_MODULE(PixelShuffle);

} // namespace nn
} // namespace torch
