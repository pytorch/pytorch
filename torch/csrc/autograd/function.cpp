#include "function.h"

#include <string>
#include <THPP/THPP.h>

#include "variable.h"
#include "torch/csrc/utils/auto_gil.h"

namespace torch { namespace autograd {

auto Function::flags(const variable_list& inputs) -> FunctionFlags {
  int num_inputs = inputs.size();
  FunctionFlags f;
  f.is_executable = false;
  f.is_volatile = false;
  f.next_functions.resize(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    auto& var = inputs[i];
    if (var) {
      f.is_executable |= var->requires_grad;
      f.is_volatile |= var->is_volatile;
      if (var->grad_fn) {
        f.next_functions[i] = std::make_pair<>(var->grad_fn, var->output_nr);
      } else {
        f.next_functions[i] = std::make_pair<>(var->get_grad_accumulator(), 0);
      }
    }
  }
  f.is_executable &= !f.is_volatile;
  return f;
}

auto Function::name() -> std::string {
  return std::string(typeid(*this).name());
}

void FunctionDeleter::operator()(Function* p) const {
    // If a wrapper exist, it owns p so we just release the refcount we hold to the PyObject
    // Otherwise, free the Function that is not used anymore
    if (p->pyobj) {
      AutoGIL gil;
      Py_DECREF(p->pyobj);
    } else {
      delete p;
    }
  }

}} // namespace torch::autograd
