#include "torch/csrc/jit/variable_flags.h"
#include "torch/csrc/autograd/variable.h"

using torch::autograd::Variable;

namespace torch { namespace jit {

// These definitions require Variable struct to be defined, so they can't be
// in tracer_state.h
VariableFlags VariableFlags::of(const Variable& var) {
  VariableFlags f;
  f.defined = var.defined();
  f.requires_grad = f.defined && var.requires_grad();
  return f;
}

}}
