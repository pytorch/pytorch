#include <torch/nn/modules/functional.h>

#include <functional>
#include <utility>

namespace torch { namespace nn {

Functional::Functional(Function function) : function_(std::move(function)) {}

Functional::Functional(std::function<Variable(Variable)> function)
    : function_([function](std::vector<Variable> input) {
        return std::vector<Variable>({function(input.front())});
      }) {}

void Functional::reset() {}

std::vector<Variable> Functional::forward(std::vector<Variable> input) {
  return function_(input);
}
}} // namespace torch::nn
