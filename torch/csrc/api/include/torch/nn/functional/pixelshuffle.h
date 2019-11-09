#pragma once

#include <torch/nn/options/pixelshuffle.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor pixel_shuffle(
    const Tensor& input,
    int64_t upscale_factor) {
  return torch::pixel_shuffle(
    input,
    upscale_factor
  );
}
} // namespace detail

inline Tensor pixel_shuffle(
    const Tensor& input,
    PixelShuffleFuncOptions options) {
  return detail::pixel_shuffle(input, options.upscale_factor());
}

} // namespace functional
} // namespace nn
} // namespace torch
