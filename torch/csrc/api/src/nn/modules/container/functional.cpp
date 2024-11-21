#include <torch/nn/modules/container/functional.h>

#include <torch/types.h>

#include <functional>
#include <utility>

namespace torch::nn {
FunctionalImpl::FunctionalImpl(Function function)
    : function_(std::move(function)) {}

void FunctionalImpl::reset() {}

void FunctionalImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Functional()";
}

Tensor FunctionalImpl::forward(Tensor input) {
  return function_(std::move(input));
}

Tensor FunctionalImpl::operator()(Tensor input) {
  return forward(std::move(input));
}

bool FunctionalImpl::is_serializable() const {
  return false;
}
} // namespace torch::nn
