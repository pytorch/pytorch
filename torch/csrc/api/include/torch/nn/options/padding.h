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

// ============================================================================

/// Options for a `D`-dimensional ReflectionPad module.
template <size_t D>
struct TORCH_API ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// If it is `int`, uses the same padding in all boundaries.
  /// If it is a 2-`tuple` (for ReflectionPad1d), uses (padding_left, padding_right).
  /// If it is a 4-`tuple` (for ReflectionPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  TORCH_ARG(ExpandingArray<D*2>, padding);
};

/// `ReflectionPadOptions` specialized for 1-D ReflectionPad.
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

/// `ReflectionPadOptions` specialized for 2-D ReflectionPad.
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

} // namespace nn
} // namespace torch
