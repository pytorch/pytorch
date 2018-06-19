#include <torch/nn/modules/functional.h>

#include <torch/tensor.h>

#include <functional>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
FunctionalImpl::FunctionalImpl(Function function)
    : function_(std::move(function)) {}

FunctionalImpl::FunctionalImpl(std::function<Variable(Variable)> function)
    : function_([function](std::vector<Variable> input) {
        return std::vector<Variable>({function(input.front())});
      }) {}

void FunctionalImpl::reset() {}

std::vector<Variable> FunctionalImpl::forward(std::vector<Variable> input) {
  return function_(input);
}
} // namespace nn
} // namespace torch
