#pragma once

#include <torch/nn/options/pixelshuffle.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor pixel_shuffle(
    const Tensor& input,
    const PixelShuffleFuncOptions& options) {
  return torch::pixel_shuffle(
    input,
    options.upscale_factor()
  );
}

} // namespace functional
} // namespace nn
} // namespace torch
