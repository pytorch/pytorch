#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `PixelShuffle` module.
///
/// Example:
/// ```
/// PixelShuffle model(PixelShuffleOptions(5));
/// ```
struct TORCH_API PixelShuffleOptions {
  PixelShuffleOptions(int64_t upscale_factor)
      : upscale_factor_(upscale_factor) {}

  /// Factor to increase spatial resolution by
  TORCH_ARG(int64_t, upscale_factor);
};

namespace functional {
/// Options for `torch::nn::functional::pixel_shuffle`.
///
/// See the documentation for `torch::nn::PixelShuffleOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pixel_shuffle(x, F::PixelShuffleFuncOptions(2));
/// ```
using PixelShuffleFuncOptions = PixelShuffleOptions;
} // namespace functional

} // namespace nn
} // namespace torch
