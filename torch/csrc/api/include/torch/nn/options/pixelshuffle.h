#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/nn/options/common.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the PixelShuffle module.
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

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(PixelShuffle, PixelShuffleFuncOptions)

} // namespace nn
} // namespace torch
