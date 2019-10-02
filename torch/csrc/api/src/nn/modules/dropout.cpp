#include <torch/nn/modules/dropout.h>

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <vector>

namespace torch {
namespace nn {
namespace detail {
template <typename Derived>
DropoutImplBase<Derived>::DropoutImplBase(const DropoutOptions& options_)
    : options(options_) {
  TORCH_CHECK(options.rate() >= 0, "Dropout rate must not be less than zero");
  TORCH_CHECK(options.rate() <= 1, "Dropout rate must not be greater than one");
}

template <typename Derived>
void DropoutImplBase<Derived>::reset() {}

template class DropoutImplBase<DropoutImpl>;
template class DropoutImplBase<FeatureDropoutImpl>;
} // namespace detail

DropoutImpl::DropoutImpl(const DropoutOptions& options_) : DropoutImplBase(options_) {}

Tensor DropoutImpl::forward(const Tensor& input) {
  return torch::dropout(input, options.rate(), this->is_training());
}

void DropoutImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Dropout(rate=" << options.rate() << ")";
}

FeatureDropoutImpl::FeatureDropoutImpl(const DropoutOptions& options_)
    : DropoutImplBase(options_) {}

Tensor FeatureDropoutImpl::forward(const Tensor& input) {
  return torch::feature_dropout(input, options.rate(), this->is_training());
}

void FeatureDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::FeatureDropout(rate=" << options.rate() << ")";
}
} // namespace nn
} // namespace torch
