#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/pixelshuffle.h>
#include <torch/nn/options/pixelshuffle.h>

#include <torch/csrc/Export.h>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelShuffle
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
/// to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an
/// upscale factor. See
/// https://pytorch.org/docs/master/nn.html#torch.nn.PixelShuffle to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelShuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```
struct TORCH_API PixelShuffleImpl
    : public torch::nn::Cloneable<PixelShuffleImpl> {
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
/// provides, and examples of how to use `PixelShuffle` with
/// `torch::nn::PixelShuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(PixelShuffle);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PixelUnshuffle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Reverses the PixelShuffle operation by rearranging elements in a tensor of
/// shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape :math:`(*,
/// C \times r^2, H, W)`, where r is a downscale factor. See
/// https://pytorch.org/docs/master/nn.html#torch.nn.PixelUnshuffle to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PixelUnshuffleOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PixelUnshuffle model(PixelUnshuffleOptions(5));
/// ```
struct TORCH_API PixelUnshuffleImpl
    : public torch::nn::Cloneable<PixelUnshuffleImpl> {
  explicit PixelUnshuffleImpl(const PixelUnshuffleOptions& options_);

  /// Pretty prints the `PixelUnshuffle` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  void reset() override;

  /// The options with which this `Module` was constructed.
  PixelUnshuffleOptions options;
};

/// A `ModuleHolder` subclass for `PixelUnshuffleImpl`.
/// See the documentation for `PixelUnshuffleImpl` class to learn what methods
/// it provides, and examples of how to use `PixelUnshuffle` with
/// `torch::nn::PixelUnshuffleOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(PixelUnshuffle);

} // namespace nn
} // namespace torch
