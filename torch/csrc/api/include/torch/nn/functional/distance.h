#pragma once

#include <torch/nn/options/distance.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    const CosineSimilarityOptions& options) {
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
    const PairwiseDistanceOptions& options) {
  return torch::pairwise_distance(
      x1,
      x2,
      options.p(),
      options.eps(),
      options.keepdim());
}

} // namespace functional
} // namespace nn
} // namespace torch
