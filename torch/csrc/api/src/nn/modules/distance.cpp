#include <torch/nn/modules/distance.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

CosineSimilarityImpl::CosineSimilarityImpl(CosineSimilarityOptions options)
    : options(std::move(options)) {}

void CosineSimilarityImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CosineSimilarity()";
}

Tensor CosineSimilarityImpl::forward(const Tensor& x1, const Tensor& x2) {
  return F::cosine_similarity(x1, x2, options);
}

// ============================================================================

PairwiseDistanceImpl::PairwiseDistanceImpl(PairwiseDistanceOptions options)
    : options(std::move(options)) {}

void PairwiseDistanceImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PairwiseDistance()";
}

Tensor PairwiseDistanceImpl::forward(const Tensor& x1, const Tensor& x2) {
  return F::pairwise_distance(x1, x2, options);
}

} // namespace nn
} // namespace torch
