#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/nn/options/common.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `CosineSimilarity` module.
struct TORCH_API CosineSimilarityOptions {
  /// Dimension where cosine similarity is computed. Default: 1
  TORCH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-8
  TORCH_ARG(double, eps) = 1e-8;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(CosineSimilarity, CosineSimilarityFuncOptions)

// ============================================================================

/// Options for the `PairwiseDistance` module.
struct TORCH_API PairwiseDistanceOptions {
  /// The norm degree. Default: 2
  TORCH_ARG(double, p) = 2.0;
  /// Small value to avoid division by zero. Default: 1e-6
  TORCH_ARG(double, eps) = 1e-6;
  /// Determines whether or not to keep the vector dimension. Default: false
  TORCH_ARG(bool, keepdim) = false;
};

TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(PairwiseDistance, PairwiseDistanceFuncOptions)

} // namespace nn
} // namespace torch
