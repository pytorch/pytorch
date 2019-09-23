#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional reflection pad functional and module.
template <size_t D>
struct ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D> padding) : padding_(padding) {}

  /// the size of the padding.
  TORCH_ARG(ExpandingArray<D>, padding);
};

/// `ReflectionPadOptions` specialized for 1-D reflection pad .
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

} // namespace nn
} // namespace torch
