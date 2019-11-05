#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `Linear` module.
struct TORCH_API LinearOptions {
  LinearOptions(int64_t in_features, int64_t out_features);
  /// size of each input sample
  TORCH_ARG(int64_t, in_features);

  /// size of each output sample
  TORCH_ARG(int64_t, out_features);

  /// If set to false, the layer will not learn an additive bias. Default: true
  TORCH_ARG(bool, bias) = true;
};

// ============================================================================

/// Options for the `Flatten` module.
struct TORCH_API FlattenOptions {
  /// first dim to flatten
  TORCH_ARG(int64_t, start_dim) = 1;
  /// last dim to flatten
  TORCH_ARG(int64_t, end_dim) = -1;
};

// ============================================================================

/// Options for the `Bilinear` module.
struct TORCH_API BilinearOptions {
  BilinearOptions(int64_t in1_features, int64_t in2_features, int64_t out_features);
  /// The number of features in input 1 (columns of the input1 matrix).
  TORCH_ARG(int64_t, in1_features);
  /// The number of features in input 2 (columns of the input2 matrix).
  TORCH_ARG(int64_t, in2_features);
  /// The number of output features to produce (columns of the output matrix).
  TORCH_ARG(int64_t, out_features);
  /// Whether to learn and add a bias after the bilinear transformation.
  TORCH_ARG(bool, bias) = true;
};

} // namespace nn
} // namespace torch
