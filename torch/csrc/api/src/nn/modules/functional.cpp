#include <torch/nn/modules/functional.h>

namespace torch { namespace nn {
Functional::Functional(std::function<variable_list(variable_list)> fun)
    : fun_(std::move(fun)) {}
Functional::Functional(std::function<Variable(Variable)> fun)
    : fun_([fun](variable_list input) {
        return variable_list({fun(input[0])});
      }) {}

variable_list Functional::forward(variable_list input) {
  return fun_(input);
}
}} // namespace torch::nn
