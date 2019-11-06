#pragma once

#include <torch/nn/options/distance.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    const CosineSimilarityFuncOptions& options) {
  return torch::cosine_similarity(
      x1,
      x2,
      options.dim(),
      options.eps());
}

// ============================================================================

inline Tensor pairwise_distance(
    const Tensor& x1,
    const Tensor& x2,
    const PairwiseDistanceFuncOptions& options) {
  return torch::pairwise_distance(
      x1,
      x2,
      options.p(),
      options.eps(),
      options.keepdim());
}

// ============================================================================

/// Computes the p-norm distance between every pair of row vectors in the input.
/// This function will be faster if the rows are contiguous.
inline Tensor pdist(const Tensor& input, double p = 2.0) {
  return torch::pdist(input, p);
}

} // namespace functional
} // namespace nn
} // namespace torch
