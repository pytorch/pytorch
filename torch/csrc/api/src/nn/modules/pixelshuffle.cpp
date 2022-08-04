#include <torch/nn/modules/pixelshuffle.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

PixelShuffleImpl::PixelShuffleImpl(const PixelShuffleOptions& options_)
    : options(options_) {}

void PixelShuffleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PixelShuffle(upscale_factor="
         << options.upscale_factor() << ")";
}

void PixelShuffleImpl::reset() {}

Tensor PixelShuffleImpl::forward(const Tensor& input) {
  return F::detail::pixel_shuffle(input, options.upscale_factor());
}

PixelUnshuffleImpl::PixelUnshuffleImpl(const PixelUnshuffleOptions& options_)
    : options(options_) {}

void PixelUnshuffleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PixelUnshuffle(downscale_factor="
         << options.downscale_factor() << ")";
}

void PixelUnshuffleImpl::reset() {}

Tensor PixelUnshuffleImpl::forward(const Tensor& input) {
  return F::detail::pixel_unshuffle(input, options.downscale_factor());
}

} // namespace nn
} // namespace torch
