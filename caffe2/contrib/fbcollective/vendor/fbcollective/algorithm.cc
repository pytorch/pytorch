#include "fbcollective/algorithm.h"

#include "fbcollective/common/logging.h"

namespace fbcollective {

Algorithm::Algorithm(const std::shared_ptr<Context>& context)
    : context_(context),
      contextRank_(context_->rank_),
      contextSize_(context_->size_) {}

// Have to provide implementation for pure virtual destructor.
Algorithm::~Algorithm() {}

std::unique_ptr<transport::Pair>& Algorithm::getPair(int i) {
  return context_->getPair(i);
}

// Helper for ring algorithms
std::unique_ptr<transport::Pair>& Algorithm::getLeftPair() {
  auto rank = (context_->size_ + context_->rank_ - 1) % context_->size_;
  FBC_ENFORCE(context_->getPair(rank), "pair missing (index ", rank, ")");
  return context_->getPair(rank);
}

// Helper for ring algorithms
std::unique_ptr<transport::Pair>& Algorithm::getRightPair() {
  auto rank = (context_->rank_ + 1) % context_->size_;
  FBC_ENFORCE(context_->getPair(rank), "pair missing (index ", rank, ")");
  return context_->getPair(rank);
}

} // namespace fbcollective
