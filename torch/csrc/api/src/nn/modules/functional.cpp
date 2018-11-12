#include <torch/nn/modules/functional.h>

#include <torch/types.h>

#include <functional>
#include <utility>

namespace torch {
namespace nn {
FunctionalImpl::FunctionalImpl(Function function)
    : function_(std::move(function)) {}

void FunctionalImpl::reset() {}

Tensor FunctionalImpl::forward(Tensor input) {
  return function_(std::move(input));
}

Tensor FunctionalImpl::operator()(Tensor input) {
  return forward(std::move(input));
}
} // namespace nn
} // namespace torch
