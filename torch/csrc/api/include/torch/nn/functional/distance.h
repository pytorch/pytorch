#pragma once

#include <torch/nn/options/distance.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor cosine_similarity(
    const Tensor& x1,
    const Tensor& x2,
    int64_t dim,
    double eps) {
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
    const CosineSimilarityFuncOptions& options = {}) {
  return detail::cosine_similarity(x1, x2, options.dim(), options.eps());
}

// ============================================================================

namespace detail {
inline Tensor pairwise_distance(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    double eps,
    bool keepdim) {
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
    const PairwiseDistanceFuncOptions& options = {}) {
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
