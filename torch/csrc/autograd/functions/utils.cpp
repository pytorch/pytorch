#include "torch/csrc/autograd/functions/utils.h"

namespace torch { namespace autograd {

variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr) {
  auto flags = Function::flags(inputs);
  variable_list result;
  result.reserve(outputs.size());
  if (flags.is_volatile) {
    for (auto& output : outputs) {
     result.emplace_back(Variable::of(std::move(output), true));
    }
  } else {
    auto grad_fn = ctr(std::move(flags));
    for (auto& output : outputs) {
      if (output) {
        result.emplace_back(std::make_shared<Variable>(std::move(output), grad_fn));
      } else {
        result.emplace_back(nullptr);
      }
    }
  }
  return result;
}

}}
