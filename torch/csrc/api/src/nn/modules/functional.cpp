#include <torch/nn/modules/functional.h>

#include <torch/tensor.h>

#include <functional>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
FunctionalImpl::FunctionalImpl(Function function)
    : function_(std::move(function)) {}

FunctionalImpl::FunctionalImpl(std::function<Tensor(Tensor)> function)
    : function_([function](std::vector<Tensor> input) {
        return std::vector<Tensor>({function(input.front())});
      }) {}

void FunctionalImpl::reset() {}

std::vector<Tensor> FunctionalImpl::forward(std::vector<Tensor> input) {
  return function_(input);
}
} // namespace nn
} // namespace torch
