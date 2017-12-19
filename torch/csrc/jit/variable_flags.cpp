#include "torch/csrc/jit/variable_flags.h"
#include "torch/csrc/autograd/variable.h"

using torch::autograd::Variable;

namespace torch { namespace jit {

// These definitions require Variable struct to be defined, so they can't be
// in tracer_state.h
VariableFlags VariableFlags::of(const Variable& var) {
  VariableFlags f;
  if (var.defined()) {
    f.was_null = false;
    f.requires_grad = var.requires_grad();
  } else {
    f.was_null = true;
  }
  return f;
}

bool VariableFlags::verify(const Variable& var) const {
  if (!var.defined()) return was_null;
  return !was_null && requires_grad == var.requires_grad();
}


}}
