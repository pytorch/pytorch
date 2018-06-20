#include <torch/nn/modules/dropout.h>

#include <torch/functions.h>
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
std::vector<Variable> DropoutImplBase<Derived>::forward(
    std::vector<Variable> input) {
  if (options_.rate_ == 0 || !this->is_training()) {
    return input;
  }
  std::vector<Variable> output;
  for (const auto& value : input) {
    const auto noise = (noise_mask(value).uniform_(0, 1) > options_.rate_)
                           .toType(value.type().scalarType())
                           .mul_(1.0f / (1.0f - options_.rate_));
    output.push_back(value * noise);
  }
  return output;
}

template <typename Derived>
const DropoutOptions& DropoutImplBase<Derived>::options() const noexcept {
  return options_;
}

template class DropoutImplBase<DropoutImpl>;
template class DropoutImplBase<Dropout2dImpl>;
} // namespace detail

DropoutOptions::DropoutOptions(double rate) : rate_(rate) {}

Variable DropoutImpl::noise_mask(Variable input) const {
  return at::empty_like(input);
}

Variable Dropout2dImpl::noise_mask(Variable input) const {
  return torch::empty({input.size(0), input.size(1), 1, 1}, input.options());
}
} // namespace nn
} // namespace torch
