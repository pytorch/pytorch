#include <torch/nn/modules/distance.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

CosineSimilarityImpl::CosineSimilarityImpl(const CosineSimilarityOptions& options_)
    : options(options_) {}

void CosineSimilarityImpl::reset() {}

void CosineSimilarityImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::CosineSimilarity"
         << "(dim=" << options.dim()
         << ", eps=" << options.eps() << ")";
}

Tensor CosineSimilarityImpl::forward(const Tensor& x1, const Tensor& x2) {
  return F::cosine_similarity(x1, x2, options);
}

// ============================================================================

PairwiseDistanceImpl::PairwiseDistanceImpl(const PairwiseDistanceOptions& options_)
    : options(options_) {}

void PairwiseDistanceImpl::reset() {}

void PairwiseDistanceImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::PairwiseDistance"
         << "(p=" << options.p()
         << ", eps=" << options.eps()
         << ", keepdim=" << options.keepdim() << ")";
}

Tensor PairwiseDistanceImpl::forward(const Tensor& x1, const Tensor& x2) {
  return F::pairwise_distance(x1, x2, options);
}

} // namespace nn
} // namespace torch
