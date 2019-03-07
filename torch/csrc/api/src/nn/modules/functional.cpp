#include <torch/nn/modules/functional.h>

#include <torch/types.h>

#include <functional>
#include <utility>

namespace torch {
namespace nn {
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
} // namespace nn
} // namespace torch
