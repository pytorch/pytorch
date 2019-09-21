#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a pad functional.
struct TORCH_API PadOptions {
  PadOptions(IntArrayRef pad) : pad_(pad) {}

  /// m-elements tuple, where m/2 <= input dimensions and m is even.
  TORCH_ARG(IntArrayRef, pad);

  /// "constant", "reflect", "replicate" or "circular". Default: "constant"
  TORCH_ARG(std::string, mode) = "constant";

  /// fill value for "constant" padding. Default: 0
  TORCH_ARG(double, value) = 0;
};

} // namespace nn
} // namespace torch
