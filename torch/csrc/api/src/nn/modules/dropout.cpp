#include <torch/nn/modules/dropout.h>

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

using AlphaDropoutOptions = DropoutOptions;

namespace detail {
template <typename Derived>
DropoutImplBase<Derived>::DropoutImplBase(const DropoutOptions& options_)
    : options(options_) {
  TORCH_CHECK(
    options.p() >= 0 && options.p() <= 1,
    "dropout probability has to be between 0 and 1, but got ", options.p()
  );
}

template <typename Derived>
void DropoutImplBase<Derived>::reset() {}

template class DropoutImplBase<DropoutImpl>;
template class DropoutImplBase<FeatureDropoutImpl>;
template class DropoutImplBase<AlphaDropoutImpl>;
} // namespace detail

DropoutImpl::DropoutImpl(const DropoutOptions& options_) : DropoutImplBase(options_) {}

Tensor DropoutImpl::forward(Tensor input) {
  return torch::dropout(input, options.p(), this->is_training());
}

void DropoutImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Dropout(p=" << options.p();
  stream << ", inplace=" << std::boolalpha << options.inplace() << ")";
}

FeatureDropoutImpl::FeatureDropoutImpl(const DropoutOptions& options_)
    : DropoutImplBase(options_) {}

Tensor FeatureDropoutImpl::forward(Tensor input) {
  return torch::feature_dropout(input, options.p(), this->is_training());
}

void FeatureDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::FeatureDropout(p=" << options.p(); 
  stream << ", inplace=" << std::boolalpha << options.inplace() << ")";
}

AlphaDropoutImpl::AlphaDropoutImpl(const AlphaDropoutOptions& options_)
    : DropoutImplBase(options_) {}

Tensor AlphaDropoutImpl::forward(Tensor input) {
  return F::alpha_dropout(input, options.p(), this->is_training());
}

void AlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AlphaDropout(p=" << options.p();
  stream << ", inplace=" << std::boolalpha << options.inplace() << ")";
}
} // namespace nn
} // namespace torch
