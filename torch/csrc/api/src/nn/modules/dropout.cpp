#include <torch/nn/modules/dropout.h>

#include <torch/tensor.h>

#include <ATen/Error.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
namespace detail {
template <typename Derived>
DropoutImplBase<Derived>::DropoutImplBase(DropoutOptions options)
    : options_(options) {
  AT_CHECK(options_.rate_ >= 0, "Dropout rate must not be less than zero");
  AT_CHECK(options_.rate_ <= 1, "Dropout rate must not be greater than one");
}

template <typename Derived>
void DropoutImplBase<Derived>::reset() {}

template <typename Derived>
Tensor DropoutImplBase<Derived>::forward(Tensor input) {
  if (options_.rate_ == 0 || !this->is_training()) {
    return input;
  }

  auto scale = 1.0f / (1.0f - options_.rate_);
  auto boolean_mask = noise_mask(input).uniform_(0, 1) > options_.rate_;
  auto noise = boolean_mask.to(input.dtype()).mul_(scale);

  return input * noise;
}

template <typename Derived>
const DropoutOptions& DropoutImplBase<Derived>::options() const noexcept {
  return options_;
}

template class DropoutImplBase<DropoutImpl>;
template class DropoutImplBase<Dropout2dImpl>;
} // namespace detail

DropoutOptions::DropoutOptions(double rate) : rate_(rate) {}

Tensor DropoutImpl::noise_mask(Tensor input) const {
  return torch::empty_like(input);
}

Tensor Dropout2dImpl::noise_mask(Tensor input) const {
  return torch::empty({input.size(0), input.size(1), 1, 1}, input.options());
}
} // namespace nn
} // namespace torch
