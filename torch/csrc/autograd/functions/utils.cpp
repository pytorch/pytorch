#include "torch/csrc/autograd/functions/utils.h"

#include <sstream>

namespace torch { namespace autograd {

variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr) {
  auto flags = Function::flags(inputs);
  variable_list result;
  result.reserve(outputs.size());
  if (flags.is_volatile) {
    for (auto& output : outputs) {
     result.emplace_back(Variable::of(output, true));
    }
  } else {
    auto grad_fn = ctr(std::move(flags));
    for (auto& output : outputs) {
      if (output.defined()) {
        result.emplace_back(std::make_shared<Variable>(output, grad_fn));
      } else {
        ++grad_fn->num_inputs;
        result.emplace_back(nullptr);
      }
    }
  }
  return result;
}

void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args) {
  if (required_args == -1) {
    required_args = args;
  }
  if (inputs.size() != (size_t)args) {
    std::stringstream ss;
    ss << name << ": expected " << args << " arguments (got " << inputs.size();
    ss << ")";
    throw std::runtime_error(ss.str());
  }
  for (int i = 0; i < required_args; ++i) {
    if (!inputs[i]) {
      std::stringstream ss;
      ss << name << ": expected Variable at argument " << i << " (got None)";
      throw std::runtime_error(ss.str());
    }
  }
}

}}
