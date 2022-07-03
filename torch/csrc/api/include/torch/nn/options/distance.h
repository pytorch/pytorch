#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `CosineSimilarity` module.
///
/// Example:
/// ```
/// CosineSimilarity model(CosineSimilarityOptions().dim(0).eps(0.5));
/// ```
struct TORCH_API CosineSimilarityOptions {
  /// Dimension where cosine similarity is computed. Default: 1
  TORCH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-8
  TORCH_ARG(double, eps) = 1e-8;
};

namespace functional {
/// Options for `torch::nn::functional::cosine_similarity`.
///
/// See the documentation for `torch::nn::CosineSimilarityOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::cosine_similarity(input1, input2,
/// F::CosineSimilarityFuncOptions().dim(1));
/// ```
using CosineSimilarityFuncOptions = CosineSimilarityOptions;
} // namespace functional

// ============================================================================

/// Options for the `PairwiseDistance` module.
///
/// Example:
/// ```
/// PairwiseDistance
/// model(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true));
/// ```
struct TORCH_API PairwiseDistanceOptions {
  /// The norm degree. Default: 2
  TORCH_ARG(double, p) = 2.0;
  /// Small value to avoid division by zero. Default: 1e-6
  TORCH_ARG(double, eps) = 1e-6;
  /// Determines whether or not to keep the vector dimension. Default: false
  TORCH_ARG(bool, keepdim) = false;
};

namespace functional {
/// Options for `torch::nn::functional::pairwise_distance`.
///
/// See the documentation for `torch::nn::PairwiseDistanceOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pairwise_distance(input1, input2, F::PairwiseDistanceFuncOptions().p(1));
/// ```
using PairwiseDistanceFuncOptions = PairwiseDistanceOptions;
} // namespace functional

} // namespace nn
} // namespace torch
