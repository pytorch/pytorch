#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the PixelShuffle module.
struct TORCH_API PixelShuffleOptions {
  PixelShuffleOptions(int64_t upscale_factor)
      : upscale_factor_(upscale_factor) {}

  /// Specifies the reduction to apply to the output.
  TORCH_ARG(int64_t, upscale_factor);
};

} // namespace nn
} // namespace torch