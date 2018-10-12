#include <torch/nn/modules/dropout.h>

#include <torch/tensor.h>

#include <ATen/core/Error.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
namespace detail {
template <typename Derived>
DropoutImplBase<Derived>::DropoutImplBase(DropoutOptions options_)
    : options(options_) {
  AT_CHECK(options.rate_ >= 0, "Dropout rate must not be less than zero");
  AT_CHECK(options.rate_ <= 1, "Dropout rate must not be greater than one");
}

template <typename Derived>
void DropoutImplBase<Derived>::reset() {}

template class DropoutImplBase<DropoutImpl>;
template class DropoutImplBase<FeatureDropoutImpl>;
} // namespace detail

DropoutOptions::DropoutOptions(double rate) : rate_(rate) {}

Tensor DropoutImpl::forward(Tensor input) {
  return torch::dropout(input, options.rate_, this->is_training());
}

Tensor FeatureDropoutImpl::forward(Tensor input) {
  return torch::feature_dropout(input, options.rate_, this->is_training());
}
} // namespace nn
} // namespace torch
