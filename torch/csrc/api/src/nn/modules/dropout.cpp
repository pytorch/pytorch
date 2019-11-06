#include <torch/nn/modules/dropout.h>

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
DropoutImplBase<D, Derived>::DropoutImplBase(const DropoutOptionsBase<D>& options_)
    : options(options_) {
  TORCH_CHECK(options.p() >= 0 && options.p() <= 1,
  	"dropout probability has to be between 0 and 1, but got ", options.p());
}

template <size_t D, typename Derived>
void DropoutImplBase<D, Derived>::reset() {}

template <size_t D, typename Derived>
void DropoutImplBase<D, Derived>::pretty_print(std::ostream& stream) const {

  stream << "torch::nn::Dropout" << D << "d"
         << "(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

Tensor DropoutImpl::forward(const Tensor& input) {
  return F::dropout(input, options, this->is_training());
}

Tensor Dropout2dImpl::forward(const Tensor& input) {
  return F::dropout2d(input, options, this->is_training());
}

Tensor Dropout3dImpl::forward(const Tensor& input) {
  return F::dropout3d(input, options, this->is_training());
}

template class DropoutImplBase<1, DropoutImpl>;
template class DropoutImplBase<2, Dropout2dImpl>;
template class DropoutImplBase<3, Dropout3dImpl>;

template class DropoutImplBase<FeatureDropoutImpl>;

FeatureDropoutImpl::FeatureDropoutImpl(const DropoutOptionsBase& options_)
    : DropoutImplBase(options_) {}

Tensor FeatureDropoutImpl::forward(const Tensor& input) {
  return torch::feature_dropout(input, options.p(), this->is_training());
}

void FeatureDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::FeatureDropout(rate=" << options.p() << ")";
}

} // namespace nn
} // namespace torch
