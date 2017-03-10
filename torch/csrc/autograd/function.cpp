#include "function.h"

#include <string>
#include <THPP/THPP.h>

#include "variable.h"

namespace torch { namespace autograd {

auto Function::flags(const variable_list& inputs) -> FunctionFlags {
  int num_inputs = inputs.size();
  FunctionFlags f;
  f.requires_grad = false;
  f.is_volatile = false;
  f.previous_functions.resize(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    auto& var = inputs[i];
    if (var) {
      f.requires_grad |= var->requires_grad;
      f.is_volatile |= var->is_volatile;
      if (var->creator) {
        f.previous_functions[i] = std::make_pair<>(var->creator, var->output_nr);
      } else {
        f.previous_functions[i] = std::make_pair<>(var, 0);
      }
    }
  }
  f.requires_grad &= !f.is_volatile;
  return f;
}

auto Function::name() -> std::string {
  return std::string(typeid(*this).name());
}

}} // namespace torch::autograd
