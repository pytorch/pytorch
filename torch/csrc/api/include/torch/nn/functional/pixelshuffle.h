#pragma once

#include <torch/nn/options/pixelshuffle.h>

namespace torch::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor pixel_shuffle(const Tensor& input, int64_t upscale_factor) {
  return torch::pixel_shuffle(input, upscale_factor);
}

inline Tensor pixel_unshuffle(const Tensor& input, int64_t downscale_factor) {
  return torch::pixel_unshuffle(input, downscale_factor);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.pixel_shuffle
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::PixelShuffleFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pixel_shuffle(x, F::PixelShuffleFuncOptions(2));
/// ```
inline Tensor pixel_shuffle(
    const Tensor& input,
    const PixelShuffleFuncOptions& options) {
  return detail::pixel_shuffle(input, options.upscale_factor());
}

inline Tensor pixel_unshuffle(
    const Tensor& input,
    const PixelUnshuffleFuncOptions& options) {
  return detail::pixel_unshuffle(input, options.downscale_factor());
}

} // namespace torch::nn::functional
