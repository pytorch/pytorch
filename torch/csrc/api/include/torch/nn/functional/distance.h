#pragma once

#include <torch/nn/options/distance.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    int64_t dim = 1,
    double eps = 1e-8) {
  return torch::cosine_similarity(
      x1,
      x2,
      dim,
      eps);
}
} // namespace detail

inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    CosineSimilarityFuncOptions options = {}) {
  return detail::cosine_similarity(x1, x2, options.dim(), options.eps());
}

// ============================================================================

namespace detail {
inline Tensor pairwise_distance(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    double eps = 1e-6,
    bool keepdim = false) {
  return torch::pairwise_distance(
      x1,
      x2,
      p,
      eps,
      keepdim);
}
} // namespace detail

inline Tensor pairwise_distance(
    const Tensor& x1,
    const Tensor& x2,
    PairwiseDistanceFuncOptions options = {}) {
  return detail::pairwise_distance(x1, x2, options.p(), options.eps(), options.keepdim());
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
