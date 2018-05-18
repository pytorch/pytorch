#include <torch/nn/modules/functional.h>

#include <functional>
#include <utility>

namespace torch { namespace nn {

Functional::Functional(Function function) : function_(std::move(function)) {}

Functional::Functional(std::function<Variable(Variable)> function)
    : function_([function](variable_list input) {
        return variable_list({function(input.front())});
      }) {}

void Functional::reset() {}

variable_list Functional::forward(variable_list input) {
  return function_(input);
}
}} // namespace torch::nn
